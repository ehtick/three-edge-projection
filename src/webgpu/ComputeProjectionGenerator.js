import { StorageBufferAttribute } from 'three/webgpu';
import { storage, uniform } from 'three/tsl';
import { getAllMeshes } from '../utils/getAllMeshes.js';
import { EdgeGenerator } from '../EdgeGenerator.js';
import { isYProjectedLineDegenerate } from '../utils/triangleLineUtils.js';
import { ProjectionGeneratorBVHComputeData } from './ProjectionGeneratorBVHComputeData.js';
import { edgeStruct, triEdgePairStruct, overlapRecordStruct } from './nodes/structs.wgsl.js';
import { EdgePairsKernel } from './EdgePairsKernel.js';
import { EdgeOverlapsKernel } from './EdgeOverlapsKernel.js';
import { overlapsToLines } from '../utils/overlapUtils.js';
import { ProjectionResult } from '../ProjectionGenerator.js';

// TODO: edge splitting — long edges that span a large portion of the scene create heavy single-thread
// work in kernel 2. splitting them into sub-edges at pack time distributes that work across more
// threads. each sub-edge would carry tStart/tEnd (its [0,1] range within the original edge) so the
// GPU overlap t0/t1 values can be remapped back to original-edge space on the CPU before merging.

export class ComputeProjectionGenerator {

	constructor( renderer ) {

		this.renderer = renderer;
		this.angleThreshold = 50;
		this.includeIntersectionEdges = true;
		this.clipY = null;
		this.edgeBatchCapacity = 1000000;

	}

	async generate( scene ) {

		const { renderer, angleThreshold, includeIntersectionEdges, clipY, edgeBatchCapacity } = this;

		// 1. collect all relevant meshes in the provided scene hierarchy
		const meshes = getAllMeshes( scene );

		// 2. compute the candidate edges accounting for clip planes and storing a map of
		// lines to original meshes
		const edgeGenerator = new EdgeGenerator();
		edgeGenerator.thresholdAngle = angleThreshold;
		edgeGenerator.clipY = clipY;

		let edges = [];
		edgeGenerator.getEdges( scene, edges );
		if ( includeIntersectionEdges ) {

			edgeGenerator.getIntersectionEdges( scene, edges );

		}

		edges = edges.filter( e => ! isYProjectedLineDegenerate( e ) );
		if ( edges.length === 0 ) return new ProjectionResult();

		// 3. construct an object bvh from the meshes and pack into a compute buffer.
		// ProjectionGeneratorBVHComputeData auto-generates missing BVHs.
		const bvhComputeData = new ProjectionGeneratorBVHComputeData( meshes );
		bvhComputeData.update();

		// allocate the edge storage buffer (sized to the largest batch we'll need)
		const batchCapacity = Math.min( edgeBatchCapacity, edges.length );
		const edgeBufferData = new Float32Array( batchCapacity * edgeStruct.getLength() );
		const edgeBufferDataU32 = new Uint32Array( edgeBufferData.buffer );
		const edgeBufferAttribute = new StorageBufferAttribute( edgeBufferData, edgeStruct.getLength() );
		const edgeStorage = storage( edgeBufferAttribute, edgeStruct ).toReadOnly().setName( 'edges' );

		// 4. allocate the pairs buffer and associated counters.
		//    worst-case sizing: assume up to 64 triangle/edge pairs per edge on average.
		const pairsCapacity = batchCapacity * 64;
		const pairsCapacityUniform = uniform( pairsCapacity );

		// pair counts buffer: 2-element array<atomic<u32>>
		//   [0] write offset  — claimed unconditionally; allows threads to detect overflow
		//   [1] dispatch count — incremented only on successful writes; safe dispatch size for K3
		const pairCountsData = new Uint32Array( 2 );
		const pairCountsAttribute = new StorageBufferAttribute( pairCountsData, 1 );
		const pairCountsStorage = storage( pairCountsAttribute, 'uint' ).toReadWrite().setName( 'pairCounts' );

		// overflow sentinel: pre-init to 0xffffffff; atomicMin from K2 records the first overflowing edgeIndex
		const overflowData = new Uint32Array( 1 );
		const overflowAttribute = new StorageBufferAttribute( overflowData, 1 );
		const overflowEdgeIndexStorage = storage( overflowAttribute, 'uint' ).toReadWrite().setName( 'overflowEdgeIndex' );

		// pairs buffer: one { edgeIndex, objectIndex, triIndex } record per qualifying pair
		const pairsData = new Uint32Array( pairsCapacity * triEdgePairStruct.getLength() );
		const pairsAttribute = new StorageBufferAttribute( pairsData, triEdgePairStruct.getLength() );
		const pairsStorage = storage( pairsAttribute, triEdgePairStruct ).toReadWrite().setName( 'pairs' );

		// build the traversal WGSL function (bakes in the four buffer nodes above)
		const traversalFn = bvhComputeData.getTraversalFn( { pairsStorage, pairCountsStorage, pairsCapacityUniform, overflowEdgeIndexStorage } );

		// overlaps buffer: one { edgeIndex, t0, t1 } record per overlap interval (at most one per pair)
		const overlapsCapacity = pairsCapacity;
		const overlapCapacityUniform = uniform( overlapsCapacity );

		const overlapCountData = new Uint32Array( 1 );
		const overlapCountAttribute = new StorageBufferAttribute( overlapCountData, 1 );
		const overlapCountStorage = storage( overlapCountAttribute, 'uint' ).toReadWrite().setName( 'overlapCount' );

		const overlapsData = new Float32Array( overlapsCapacity * overlapRecordStruct.getLength() );
		const overlapsAttribute = new StorageBufferAttribute( overlapsData, overlapRecordStruct.getLength() );
		const overlapsStorage = storage( overlapsAttribute, overlapRecordStruct ).toReadWrite().setName( 'overlaps' );

		const overlapsFn = bvhComputeData.getOverlapsFn( { pairsStorage, edgesStorage: edgeStorage, overlapsStorage, overlapCountStorage, overlapCapacityUniform } );

		// construct kernels — kernels are created once and reused across batches
		const edgePairsKernel = new EdgePairsKernel( traversalFn, edgeStorage );
		const edgeOverlapsKernel = new EdgeOverlapsKernel( overlapsFn );

		// CPU loop - advances a head pointer through the edge list until all edges are processed
		const edgeStructStride = edgeStruct.getLength();
		const collector = new ProjectionResult();
		let headIndex = 0;
		while ( headIndex < edges.length ) {

			const batchSize = Math.min( batchCapacity, edges.length - headIndex );
			const intervalsByEdge = new Map();

			// K2+K3 retry loop — handles the case where the pairs buffer overflows by re-running
			// from the first overflowing edge index until the entire batch is covered.
			// retryOffset is the local index (within this batch) of the first unprocessed edge.
			let retryOffset = 0;
			while ( retryOffset < batchSize ) {

				const currentCount = batchSize - retryOffset;

				// pack edges [retryOffset .. batchSize-1] into positions [0 .. currentCount-1]
				for ( let i = 0; i < currentCount; i ++ ) {

					const edgeIndex = headIndex + retryOffset + i;
					const { start, end } = edges[ edgeIndex ];
					const offset = i * edgeStructStride;
					start.toArray( edgeBufferData, offset );
					end.toArray( edgeBufferData, offset + 3 );
					edgeBufferDataU32[ offset + 6 ] = edgeIndex;

				}

				edgeBufferAttribute.needsUpdate = true;

				// 5. kernel 2 (pairs): one thread per edge traverses the BVH and writes
				//    { edgeIndex, objectIndex, triIndex } records. pairCounts[0] is the raw
				//    write offset; pairCounts[1] is the number of valid writes for K3 dispatch.
				pairCountsData[ 0 ] = 0;
				pairCountsData[ 1 ] = 0;
				pairCountsAttribute.needsUpdate = true;
				overflowData[ 0 ] = 0xffffffff;
				overflowAttribute.needsUpdate = true;

				edgePairsKernel.edgeCount = currentCount;
				await renderer.compute( edgePairsKernel.kernel, edgePairsKernel.getDispatchSize( currentCount ) );

				// 6. read back dispatch count for K3 and the overflow sentinel.
				const pairCountsBuf = await renderer.getArrayBufferAsync( pairCountsAttribute );
				const pairDispatchCount = new Uint32Array( pairCountsBuf )[ 1 ];

				// 7. kernel 3 (overlaps): one thread per valid pair. each thread runs the overlap
				//    computation and writes { edgeIndex, t0, t1 } to the overlaps buffer.
				if ( pairDispatchCount > 0 ) {

					overlapCountData[ 0 ] = 0;
					overlapCountAttribute.needsUpdate = true;

					edgeOverlapsKernel.pairCount = pairDispatchCount;
					await renderer.compute( edgeOverlapsKernel.kernel, edgeOverlapsKernel.getDispatchSize( pairDispatchCount ) );

					// 8. read back overlaps and group intervals by edgeIndex.
					const overlapCountBuf = await renderer.getArrayBufferAsync( overlapCountAttribute );
					const overlapCount = Math.min( new Uint32Array( overlapCountBuf )[ 0 ], overlapsCapacity );

					if ( overlapCount > 0 ) {

						const overlapsBuf = await renderer.getArrayBufferAsync( overlapsAttribute );
						const overlapsU32 = new Uint32Array( overlapsBuf );
						const overlapsF32 = new Float32Array( overlapsBuf );

						// pair.edgeIndex is a LOCAL buffer index (0..currentCount-1); add the
						// batch and retry offsets to recover the global edge index for the map key.
						const stride = overlapRecordStruct.getLength(); // 3
						for ( let i = 0; i < overlapCount; i ++ ) {

							const edgeIndex = headIndex + retryOffset + overlapsU32[ i * stride ];
							const t0 = overlapsF32[ i * stride + 1 ];
							const t1 = overlapsF32[ i * stride + 2 ];
							if ( ! intervalsByEdge.has( edgeIndex ) ) intervalsByEdge.set( edgeIndex, [] );
							intervalsByEdge.get( edgeIndex ).push( [ t0, t1 ] );

						}

					}

				}

				// check overflow: if no overflow occurred, the batch segment is fully processed.
				const overflowBuf = await renderer.getArrayBufferAsync( overflowAttribute );
				const overflowGlobalIndex = new Uint32Array( overflowBuf )[ 0 ];
				if ( overflowGlobalIndex === 0xffffffff ) break;

				// advance past the processed edges. if there's no progress (e.g. a single edge
				// exceeds the buffer capacity), skip it to avoid an infinite loop.
				const nextRetryOffset = overflowGlobalIndex - headIndex;
				if ( nextRetryOffset <= retryOffset ) break;
				retryOffset = nextRetryOffset;

			}

			// 9. sort, merge, and convert each edge's hidden intervals to visible/hidden line segments.
			//    edges absent from intervalsByEdge had no occluding triangles and are fully visible.
			for ( let i = 0; i < batchSize; i ++ ) {

				const edgeIndex = headIndex + i;
				const intervals = intervalsByEdge.get( edgeIndex ) || [];
				intervals.sort( ( a, b ) => a[ 0 ] - b[ 0 ] );
				const merged = [];
				for ( const [ t0, t1 ] of intervals ) {

					if ( merged.length === 0 || t0 > merged[ merged.length - 1 ][ 1 ] ) {

						merged.push( [ t0, t1 ] );

					} else {

						merged[ merged.length - 1 ][ 1 ] = Math.max( merged[ merged.length - 1 ][ 1 ], t1 );

					}

				}

				overlapsToLines( edges[ edgeIndex ], merged, false, collector.visibleEdges );
				overlapsToLines( edges[ edgeIndex ], merged, true, collector.hiddenEdges );

			}

			// 10. advance the head pointer for the next batch
			headIndex += batchSize;

		}

		return collector;

	}

}
