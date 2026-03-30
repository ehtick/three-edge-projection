import { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';
import { getAllMeshes } from '../utils/getAllMeshes.js';
import { EdgeGenerator } from '../EdgeGenerator.js';
import { isYProjectedLineDegenerate } from '../utils/triangleLineUtils.js';
import { ProjectionGeneratorBVHComputeData } from './ProjectionGeneratorBVHComputeData.js';
import { edgeStruct, overlapRecordStruct } from './nodes/structs.wgsl.js';
import { EdgePairsKernel } from './kernels/EdgePairsKernel.js';
import { overlapsToLines } from '../utils/overlapUtils.js';
import { insertOverlap } from '../utils/getProjectedOverlaps.js';
import { ProjectionResult } from '../ProjectionGenerator.js';
import { ZeroOutBufferKernel } from './kernels/ZeroOutBufferKernel.js';

// TODO: edge splitting — long edges that span a large portion of the scene create heavy single-thread
// work in kernel 2. splitting them into sub-edges at pack time distributes that work across more
// threads. each sub-edge would carry tStart/tEnd (its [0,1] range within the original edge) so the
// GPU overlap t0/t1 values can be remapped back to original-edge space on the CPU before merging.

// TODO: Consider storing the ranges with multiple edges clipped per thread to reduce the array size needed

const MAX_BUFFER_SIZE = 134217728;

const MAX_PAIRS_COUNT = Math.floor( MAX_BUFFER_SIZE / ( overlapRecordStruct.getLength() * 4 ) );

const nextFrame = () => new Promise( resolve => requestAnimationFrame( resolve ) );
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

		// overlap output buffer and atomic counter
		const overlapsAttribute = new IndirectStorageBufferAttribute( MAX_PAIRS_COUNT, overlapRecordStruct.getLength(), Uint32Array );
		const bufferPointersAttribute = new IndirectStorageBufferAttribute( 1, 1 );
		const overflowFlagAttribute = new IndirectStorageBufferAttribute( 1, 1 );

		const overlapsStorage = storage( overlapsAttribute, overlapRecordStruct ).setName( 'overlaps' );
		const bufferPointersStorage = storage( bufferPointersAttribute, 'uint' ).toAtomic();
		const overflowFlagStorage = storage( overflowFlagAttribute, 'uint' ).setName( 'overflowFlag' ).toAtomic();

		//

		// set up scene data
		const bvhComputeData = new ProjectionGeneratorBVHComputeData( meshes );
		bvhComputeData.update();
		bvhComputeData.fns.collectTriEdgePairs = bvhComputeData.getCollectTriEdgePairsFn( {
			overlapsStorage: overlapsStorage,
			bufferPointersStorage: bufferPointersStorage,
			overflowFlagStorage: overflowFlagStorage,
		} );

		// initialize kernels
		const edgePairsKernel = new EdgePairsKernel();
		edgePairsKernel.setWorkgroupSize( 64, 1, 1 );
		edgePairsKernel.edges = edgeBufferAttribute;
		edgePairsKernel.bvhData = bvhComputeData;

		const zeroOutKernel = new ZeroOutBufferKernel();
		zeroOutKernel.setWorkgroupSize( 1, 1, 1 );

		//
		const intervalsByEdge = new Map();
		let progress = 0;
		const promises = [];
		const edgeStructStride = edgeStruct.getLength();

		const jobQueue = new JobQueue();

		const runJob = async ( start, count ) => {

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

			// clear the overlaps counter and overflow flag
			zeroOutKernel.target = bufferPointersAttribute;
			renderer.compute( zeroOutKernel.kernel, [ 1, 1, 1 ] );

			zeroOutKernel.target = overflowFlagAttribute;
			renderer.compute( zeroOutKernel.kernel, [ 1, 1, 1 ] );

			// traverse BVH and write overlaps directly
			edgePairsKernel.edgesToProcess = count;
			renderer.compute( edgePairsKernel.kernel, edgePairsKernel.getDispatchSize( count ) );

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

					// split the job in half and re-queue both halves
					const half = Math.ceil( count / 2 );
					promises.push( jobQueue.add( runJob, [ start, half ] ) );
					promises.push( jobQueue.add( runJob, [ start + half, count - half ] ) );
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
				const ei = start + overlapsU32[ index + 0 ];
				const t0 = overlapsF32[ index + 1 ];
				const t1 = overlapsF32[ index + 2 ];

				if ( ! intervalsByEdge.has( ei ) ) {

					intervalsByEdge.set( ei, [] );

				}

				insertOverlap( [ t0, t1 ], intervalsByEdge.get( ei ) );

			}

			progress += count;

			// fire progress
			if ( onProgress ) {

				onProgress( progress / edges.length );

			}

		};

		// enqueue initial jobs
		for ( let e = 0; e < edges.length; e += batchCapacity ) {

			promises.push( jobQueue.add( runJob, [ e, Math.min( batchCapacity, edges.length - e ) ] ) );

		}

		// drain — sequential iteration naturally picks up overflow sub-jobs added to promises
		for ( let i = 0; i < promises.length; i ++ ) {

			await promises[ i ];

		}

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

class JobQueue {

	constructor() {

		this.queue = [];
		this.maxJobs = 10;
		this.currJobs = 0;
		this._scheduled = false;

	}

	add( cb, args ) {

		return new Promise( ( resolve, reject ) => {

			this.queue.push( {
				run: () => {

					const res = cb( ...args );
					res
						.then( resolve )
						.catch( reject );

					return res;

				},
				reject,
			} );
			this.scheduleRun();

		} );

	}

	cancelAll() {

		const { queue } = this;
		while ( queue.length > 0 ) {

			const entry = queue.shift();
			entry.reject( new Error( 'JobQueue: cancelled' ) );

		}

	}

	async runJobs() {

		const { queue } = this;
		while ( this.currJobs < this.maxJobs ) {

			if ( queue.length === 0 ) {

				return;

			}

			this.currJobs ++;

			await nextFrame();

			const entry = queue.shift();
			entry.run()
				.finally( () => {

					this.currJobs --;
					this.scheduleRun();

				} );

		}

	}

	scheduleRun() {

		if ( this._scheduled ) {

			return;

		}

		this._scheduled = true;
		requestAnimationFrame( async () => {

			await this.runJobs();
			this._scheduled = false;

		} );

	}

}
