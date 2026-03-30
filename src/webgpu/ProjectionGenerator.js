import { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';
import { getAllMeshes } from '../utils/getAllMeshes.js';
import { EdgeGenerator } from '../EdgeGenerator.js';
import { isYProjectedLineDegenerate } from '../utils/triangleLineUtils.js';
import { ProjectionGeneratorBVHComputeData } from './ProjectionGeneratorBVHComputeData.js';
import { edgeStruct, triEdgePairStruct, overlapRecordStruct } from './nodes/structs.wgsl.js';
import { EdgePairsKernel } from './kernels/EdgePairsKernel.js';
import { EdgeOverlapsKernel } from './kernels/EdgeOverlapsKernel.js';
import { overlapsToLines } from '../utils/overlapUtils.js';
import { insertOverlap } from '../utils/getProjectedOverlaps.js';
import { ProjectionResult } from '../ProjectionGenerator.js';
import { ZeroOutBufferKernel } from './kernels/ZeroOutBufferKernel.js';

// TODO: edge splitting — long edges that span a large portion of the scene create heavy single-thread
// work in kernel 2. splitting them into sub-edges at pack time distributes that work across more
// threads. each sub-edge would carry tStart/tEnd (its [0,1] range within the original edge) so the
// GPU overlap t0/t1 values can be remapped back to original-edge space on the CPU before merging.

// TODO: Consider storing the ranges with multiple edges clipped per thread to reduce the array size needed

const MAX_DISPATCH_SIZE = 65535;
const MAX_BUFFER_SIZE = 134217728;

const LARGEST_STRUCT_SIZE = Math.max( triEdgePairStruct.getLength(), overlapRecordStruct.getLength() );
const MAX_PAIRS_COUNT = Math.floor( MAX_BUFFER_SIZE / ( LARGEST_STRUCT_SIZE * 4 ) );

export class ProjectionGenerator {

	constructor( renderer ) {

		this.renderer = renderer;
		this.angleThreshold = 50;
		this.batchSize = 10000;
		this.includeIntersectionEdges = true;

	}

	async generate( scene, options = {} ) {

		const { renderer, angleThreshold, includeIntersectionEdges, batchSize } = this;
		const { onProgress = null, signal = null } = options;

		// collect meshes
		const meshes = getAllMeshes( scene );

		// generate edges
		const edgeGenerator = new EdgeGenerator();
		edgeGenerator.thresholdAngle = angleThreshold;

		// adjust the offset to account for floating point error in the edge processing and intersections.
		// NOTE: Ideally we should be applying this relative to the scale of the values being used rather that
		// using a fixed offset.
		edgeGenerator.yOffset = 5 * 1e-5;

		let edges = [];
		edgeGenerator.getEdges( scene, edges );
		if ( includeIntersectionEdges ) {

			edgeGenerator.getIntersectionEdges( scene, edges );

		}

		edges = edges.filter( e => ! isYProjectedLineDegenerate( e ) );

		edges.sort( ( a, b ) => {

			const uuidA = a.mesh.uuid;
			const uuidB = b.mesh.uuid;
			if ( uuidA === uuidB ) {

				return 0;

			} else {

				return uuidA < uuidB ? - 1 : 1;

			}

		} );

		if ( edges.length === 0 ) {

			return new ProjectionResult();

		}

		//

		// allocate a buffer of edges for at most the requested capacity
		const batchCapacity = Math.min( batchSize, edges.length );
		const edgeBufferData = new Float32Array( batchCapacity * edgeStruct.getLength() );
		const edgeBufferDataU32 = new Uint32Array( edgeBufferData.buffer );
		const edgeBufferAttribute = new StorageBufferAttribute( edgeBufferData, edgeStruct.getLength() );

		// store the triangle / edge pairs to
		const triEdgePairsAttribute = new StorageBufferAttribute( MAX_PAIRS_COUNT, triEdgePairStruct.getLength(), Uint32Array );
		const triEdgePairsCountAttribute = new StorageBufferAttribute( 2, 1 );
		const overlapsAttribute = new IndirectStorageBufferAttribute( MAX_PAIRS_COUNT, overlapRecordStruct.getLength(), Uint32Array );
		const bufferPointersAttribute = new IndirectStorageBufferAttribute( 2, 1 );
		const overflowFlagAttribute = new IndirectStorageBufferAttribute( 1, 1 );

		const triEdgePairsStorage = storage( triEdgePairsAttribute, triEdgePairStruct ).setName( 'TriEdge' );
		const triEdgePairsCountStorage = storage( triEdgePairsCountAttribute, 'uint' ).toAtomic();
		const overflowFlagStorage = storage( overflowFlagAttribute, 'uint' ).setName( 'overflowFlag' ).toAtomic();

		//

		// set up scene data
		const bvhComputeData = new ProjectionGeneratorBVHComputeData( meshes );
		bvhComputeData.update();
		bvhComputeData.fns.collectTriEdgePairs = bvhComputeData.getCollectTriEdgePairsFn( {
			pairsStorage: triEdgePairsStorage,
			pairsCountsStorage: triEdgePairsCountStorage,
			overflowFlagStorage: overflowFlagStorage,
		} );

		// initialize kernels
		const edgePairsKernel = new EdgePairsKernel();
		edgePairsKernel.setWorkgroupSize( 64, 1, 1 );
		edgePairsKernel.edges = edgeBufferAttribute;
		edgePairsKernel.bvhData = bvhComputeData;

		const edgeOverlapsKernel = new EdgeOverlapsKernel();
		edgeOverlapsKernel.setWorkgroupSize( 64, 1, 1 );
		edgeOverlapsKernel.pairs = triEdgePairsAttribute;
		edgeOverlapsKernel.pairsCount = triEdgePairsCountAttribute;
		edgeOverlapsKernel.bvhData = bvhComputeData;
		edgeOverlapsKernel.edges = edgeBufferAttribute;
		edgeOverlapsKernel.overlaps = overlapsAttribute;
		edgeOverlapsKernel.bufferPointers = bufferPointersAttribute;

		const zeroOutKernel = new ZeroOutBufferKernel();
		zeroOutKernel.setWorkgroupSize( 1, 1, 1 );

		//
		const intervalsByEdge = new Map();
		let progress = 0;
		const promises = [];
		const edgeStructStride = edgeStruct.getLength();

		// build initial job queue — one job per batchCapacity chunk
		const queue = [];
		for ( let e = 0; e < edges.length; e += batchCapacity ) {

			queue.push( { start: e, count: Math.min( batchCapacity, edges.length - e ) } );

		}

		while ( queue.length > 0 ) {

			const { start, count } = queue.shift();

			await new Promise( resolve => requestAnimationFrame( resolve ) );

			// run the data read back asynchronously so we can prepare and issue the subsequent
			// compute while we wait for this data.
			promises.push( ( async () => {

				if ( signal?.isAborted ) {

					return;

				}

				// fill out the edges array
				for ( let i = 0; i < count; i ++ ) {

					const edge = edges[ start + i ];
					const offset = i * edgeStructStride;
					edge.start.toArray( edgeBufferData, offset );
					edge.end.toArray( edgeBufferData, offset + 3 );
					edgeBufferDataU32[ offset + 6 ] = i;

				}

				edgeBufferAttribute.needsUpdate = true;

				// clear the pairs counts & overlaps pointers
				zeroOutKernel.target = triEdgePairsCountAttribute;
				renderer.compute( zeroOutKernel.kernel, [ 2, 1, 1 ] );

				zeroOutKernel.target = bufferPointersAttribute;
				renderer.compute( zeroOutKernel.kernel, [ 2, 1, 1 ] );

				zeroOutKernel.target = overflowFlagAttribute;
				renderer.compute( zeroOutKernel.kernel, [ 1, 1, 1 ] );

				// accumulate potential triangle-edge overlap pairs
				edgePairsKernel.edgesToProcess = count;
				renderer.compute( edgePairsKernel.kernel, edgePairsKernel.getDispatchSize( count ) );

				const dispatchSize = edgeOverlapsKernel.getDispatchSize( triEdgePairsAttribute.count )[ 0 ];
				const dispatchStepSize = Math.min( dispatchSize, MAX_DISPATCH_SIZE );

				// run dispatches for all overlaps
				for ( let i = 0; i < dispatchSize; i += dispatchStepSize ) {

					renderer.compute( edgeOverlapsKernel.kernel, [ dispatchStepSize, 1, 1 ] );

				}

				const local_start = start;
				const local_count = count;
				const [ overlaps, bufferPointers, overflowBuffer ] = await Promise.all( [
					renderer.getArrayBufferAsync( overlapsAttribute ),
					renderer.getArrayBufferAsync( bufferPointersAttribute ),
					renderer.getArrayBufferAsync( overflowFlagAttribute ),
				] );

				if ( signal?.isAborted ) {

					return;

				}

				const overflow = new Uint32Array( overflowBuffer )[ 0 ];
				if ( overflow > 0 ) {

					if ( count === 1 ) {

						console.error( `ProjectionGenerator: Overlaps buffer insufficient size to store all segments. Please report to three-edge-projection.` );

					} else {

						// split the job in half and push both halves back to the front of the queue
						const half = Math.ceil( count / 2 );
						queue.push( { start: start + half, count: count - half } );
						queue.push( { start, count: half } );
						return;

					}

				}

				// read buffers
				const overlapsF32 = new Float32Array( overlaps );
				const overlapsU32 = new Uint32Array( overlaps );
				const bufferPointersU32 = new Uint32Array( bufferPointers );
				const stride = overlapRecordStruct.getLength();

				// push the overlaps
				for ( let oi = 0, ol = bufferPointersU32[ 0 ]; oi < ol; oi ++ ) {

					const index = oi * stride;
					const ei = local_start + overlapsU32[ index + 0 ];
					const t0 = overlapsF32[ index + 1 ];
					const t1 = overlapsF32[ index + 2 ];

					if ( ! intervalsByEdge.has( ei ) ) {

						intervalsByEdge.set( ei, [] );

					}

					insertOverlap( [ t0, t1 ], intervalsByEdge.get( ei ) );

				}

				progress += local_count;

				// fire progress
				if ( onProgress ) {

					onProgress( progress / edges.length );

				}

			} )() );

		}

		// wait for all the data read back to finish
		await Promise.all( promises );

		signal?.throwIfAborted();

		// push all edges to the "results" object
		const collector = new ProjectionResult();
		for ( let i = 0; i < edges.length; i ++ ) {

			const mesh = edges[ i ].mesh;
			if ( ! collector.visibleEdges.meshToSegments.has( mesh ) ) {

				collector.visibleEdges.meshToSegments.set( mesh, [] );
				collector.hiddenEdges.meshToSegments.set( mesh, [] );

			}

			const intervals = intervalsByEdge.get( i ) || [];
			overlapsToLines( edges[ i ], intervals, false, collector.visibleEdges.meshToSegments.get( mesh ) );
			overlapsToLines( edges[ i ], intervals, true, collector.hiddenEdges.meshToSegments.get( mesh ) );

		}

		return collector;

	}

}
