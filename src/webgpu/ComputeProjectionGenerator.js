import { StorageBufferAttribute } from 'three/webgpu';
import { storage, uniform } from 'three/tsl';
import { getAllMeshes } from '../utils/getAllMeshes.js';
import { EdgeGenerator } from '../EdgeGenerator.js';
import { isYProjectedLineDegenerate } from '../utils/triangleLineUtils.js';
import { ProjectionGeneratorBVHComputeData } from './ProjectionGeneratorBVHComputeData.js';
import { edgeStruct, triEdgePairStruct, overlapRecordStruct } from './nodes/structs.wgsl.js';
import { EdgeCountKernel } from './EdgeCountKernel.js';
import { EdgePairsKernel } from './EdgePairsKernel.js';
import { EdgeOverlapsKernel } from './EdgeOverlapsKernel.js';
import { overlapsToLines } from '../utils/overlapUtils.js';

// TODO: edge splitting — long edges that span a large portion of the scene create heavy single-thread
// work in kernels 1 and 2. splitting them into sub-edges at pack time (step 4) distributes that work
// across more threads. each sub-edge would carry tStart/tEnd (its [0,1] range within the original edge)
// so the GPU overlap t0/t1 values can be remapped back to original-edge space on the CPU before merging.

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

		// 4. allocate the pairs buffer and all associated atomic counters.
		//    worst-case sizing: assume up to 64 triangle/edge pairs per edge on average.
		const pairsCapacity = batchCapacity * 64;
		const pairsCapacityUniform = uniform( pairsCapacity );

		// atomic pair counter: element 0 holds the running count of written pairs
		const pairCountData = new Uint32Array( 1 );
		const pairCountAttribute = new StorageBufferAttribute( pairCountData, 1 );
		const pairCountStorage = storage( pairCountAttribute, 'uint' ).toReadWrite().setName( 'pairCount' );

		// overflow sentinel: pre-init to 0xffffffff; atomicMin from K2 records the first overflowing edgeIndex
		const overflowData = new Uint32Array( 1 );
		const overflowAttribute = new StorageBufferAttribute( overflowData, 1 );
		const overflowEdgeIndexStorage = storage( overflowAttribute, 'uint' ).toReadWrite().setName( 'overflowEdgeIndex' );

		// pairs buffer: one { edgeIndex, objectIndex, triIndex } record per qualifying pair
		const pairsData = new Uint32Array( pairsCapacity * triEdgePairStruct.getLength() );
		const pairsAttribute = new StorageBufferAttribute( pairsData, triEdgePairStruct.getLength() );
		const pairsStorage = storage( pairsAttribute, triEdgePairStruct ).toReadWrite().setName( 'pairs' );

		// build the traversal WGSL function (bakes in the four buffer nodes above)
		const traversalFn = bvhComputeData.getTraversalFn( { pairsStorage, pairCountStorage, pairsCapacityUniform, overflowEdgeIndexStorage } );

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
		const edgeCountKernel = new EdgeCountKernel( traversalFn, edgeStorage, pairCountStorage );
		const edgePairsKernel = new EdgePairsKernel( traversalFn, edgeStorage );
		const edgeOverlapsKernel = new EdgeOverlapsKernel( overlapsFn );

		// CPU loop - advances a head pointer through the edge list until all edges are processed:
		const edgeStructStride = edgeStruct.getLength();
		const result = [];
		let headIndex = 0;
		while ( headIndex < edges.length ) {

			const batchSize = Math.min( batchCapacity, edges.length - headIndex );

			// pack the next batch of edges into the storage buffer
			for ( let i = 0; i < batchSize; i ++ ) {

				const edgeIndex = headIndex + i;
				const { start, end } = edges[ edgeIndex ];
				const offset = i * edgeStructStride;
				start.toArray( edgeBufferData, offset );
				end.toArray( edgeBufferData, offset + 3 );
				edgeBufferDataU32[ offset + 6 ] = edgeIndex;

			}

			edgeBufferAttribute.needsUpdate = true;

			// 5. kernel 1 (count): one thread per edge traverses the BVH and atomically accumulates
			//    the total number of triangle/edge pairs into pairCount[0].  the result feeds K3's
			//    indirect dispatch buffer (not yet implemented).
			pairCountData[ 0 ] = 0;
			pairCountAttribute.needsUpdate = true;

			edgeCountKernel.edgeCount = batchSize;
			await renderer.compute( edgeCountKernel.kernel, edgeCountKernel.getDispatchSize( batchSize ) );

			// 6. kernel 2 (pairs): one thread per edge traverses the BVH again and writes
			//    { edgeIndex, objectIndex, triIndex } records to the pairs buffer using atomicAdd to
			//    claim each slot. if the buffer is full the thread early-outs and records the first
			//    overflowing edgeIndex via atomicMin so the caller can retry the remaining edges.
			pairCountData[ 0 ] = 0;
			pairCountAttribute.needsUpdate = true;
			overflowData[ 0 ] = 0xffffffff;
			overflowAttribute.needsUpdate = true;

			edgePairsKernel.edgeCount = batchSize;
			await renderer.compute( edgePairsKernel.kernel, edgePairsKernel.getDispatchSize( batchSize ) );

			// 7. kernel 3 (overlaps): read back K2's pair count, then dispatch one thread per pair.
			//    each thread fetches the edge + triangle, runs the overlap computation, and writes
			//    { edgeIndex, t0, t1 } to the overlaps buffer via atomicAdd.
			const pairCountBuffer = await renderer.getArrayBufferAsync( pairCountAttribute );
			const pairCount = Math.min( new Uint32Array( pairCountBuffer )[ 0 ], pairsCapacity );

			if ( pairCount > 0 ) {

				overlapCountData[ 0 ] = 0;
				overlapCountAttribute.needsUpdate = true;

				edgeOverlapsKernel.pairCount = pairCount;
				await renderer.compute( edgeOverlapsKernel.kernel, edgeOverlapsKernel.getDispatchSize( pairCount ) );

				// 8. read back the overlaps buffer, group intervals by edgeIndex, merge overlapping
				//    [t0, t1] ranges per edge on the CPU, and convert to line segments.
				const overlapCountBuf = await renderer.getArrayBufferAsync( overlapCountAttribute );
				const overlapCount = Math.min( new Uint32Array( overlapCountBuf )[ 0 ], overlapsCapacity );

				if ( overlapCount > 0 ) {

					const overlapsBuf = await renderer.getArrayBufferAsync( overlapsAttribute );
					const overlapsU32 = new Uint32Array( overlapsBuf );
					const overlapsF32 = new Float32Array( overlapsBuf );

					// group intervals by edgeIndex
					const stride = overlapRecordStruct.getLength(); // 3
					const intervalsByEdge = new Map();
					for ( let i = 0; i < overlapCount; i ++ ) {

						const edgeIndex = overlapsU32[ i * stride ];
						const t0 = overlapsF32[ i * stride + 1 ];
						const t1 = overlapsF32[ i * stride + 2 ];
						if ( ! intervalsByEdge.has( edgeIndex ) ) intervalsByEdge.set( edgeIndex, [] );
						intervalsByEdge.get( edgeIndex ).push( [ t0, t1 ] );

					}

					// sort, merge, and convert each edge's intervals to visible line segments
					for ( const [ edgeIndex, intervals ] of intervalsByEdge ) {

						intervals.sort( ( a, b ) => a[ 0 ] - b[ 0 ] );
						const merged = [];
						for ( const [ t0, t1 ] of intervals ) {

							if ( merged.length === 0 || t0 > merged[ merged.length - 1 ][ 1 ] ) {

								merged.push( [ t0, t1 ] );

							} else {

								merged[ merged.length - 1 ][ 1 ] = Math.max( merged[ merged.length - 1 ][ 1 ], t1 );

							}

						}

						overlapsToLines( edges[ edgeIndex ], merged, false, result );

					}

				}

			}

			// TODO: overflow retry — if overflowData[0] !== 0xffffffff K2 overflowed; re-run K2+K3
			//    starting from overflowData[0] before advancing the head pointer.

			// 9. advance the head pointer for the next batch
			headIndex += batchSize;

		}

		return result;

	}

}
