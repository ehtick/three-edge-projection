/** @import { Object3D, BufferGeometry } from 'three' */
/** @import { WebGPURenderer } from 'three/webgpu' */
import { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import { sign, storage } from 'three/tsl';
import { getAllMeshes } from '../utils/getAllMeshes.js';
import { EdgeGenerator } from '../EdgeGenerator.js';
import { isYProjectedLineDegenerate } from '../utils/triangleLineUtils.js';
import { ProjectionGeneratorBVHComputeData } from './ProjectionGeneratorBVHComputeData.js';
import { edgeStruct, overlapRecordStruct } from './nodes/structs.wgsl.js';
import { EdgeOverlapsKernel } from './kernels/EdgeOverlapsKernel.js';
import { overlapsToLines } from '../utils/overlapUtils.js';
import { insertOverlap } from '../utils/getProjectedOverlaps.js';
import { ProjectionResult } from '../ProjectionGenerator.js';
import { ZeroOutBufferKernel } from './kernels/ZeroOutBufferKernel.js';
import { nextFrame } from '../utils/nextFrame.js';

// TODO: Consider storing the ranges with multiple edges clipped per thread to reduce the array size needed

const MAX_BUFFER_SIZE = 134217728;

const MAX_OVERLAPS_COUNT = Math.floor( MAX_BUFFER_SIZE / ( overlapRecordStruct.getLength() * 4 ) );

/**
 * Takes the WebGPURenderer instance used to run compute kernels.
 * @param {WebGPURenderer} renderer
 */
export class ProjectionGenerator {

	constructor( renderer ) {

		this.renderer = renderer;

		/**
		 * The threshold angle in degrees at which edges are generated.
		 * @type {number}
		 */
		this.angleThreshold = 50;

		/**
		 * The number of edges to process in one compute kernel pass. Larger values can process
		 * faster but may cause internal buffers to overflow, resulting in extra kernel executions,
		 * taking more time.
		 * @type {number}
		 */
		this.batchSize = 100000;

		/**
		 * Whether to generate edges representing the intersections between triangles.
		 * @type {boolean}
		 */
		this.includeIntersectionEdges = true;

		/**
		 * How long to spend generating edges.
		 * @type {number}
		 */
		this.iterationTime = 30;

	}

	/**
	 * Asynchronously generate the edge geometry result.
	 * @param {Object3D|BufferGeometry|Array<Object3D>} scene
	 * @param {Object} [options]
	 * @param {( percent: number, message: string ) => void} [options.onProgress]
	 * @param {AbortSignal} [options.signal]
	 * @returns {Promise<ProjectionResult>}
	 */
	async generate( scene, options = {} ) {

		const { renderer, angleThreshold, includeIntersectionEdges, batchSize, iterationTime } = this;
		const { onProgress = null, signal = null } = options;

		// collect meshes
		const meshes = getAllMeshes( scene );

		// generate edges
		const edgeGenerator = new EdgeGenerator();
		edgeGenerator.thresholdAngle = angleThreshold;
		edgeGenerator.iterationTime = iterationTime;

		// adjust the offset to account for floating point error in the edge processing and intersections.
		// NOTE: Ideally we should be applying this relative to the scale of the values being used rather that
		// using a fixed offset.
		edgeGenerator.yOffset = 5 * 1e-5;

		if ( onProgress ) {

			onProgress( 0, 'Generating Edges' );

		}

		let edges = [];
		await edgeGenerator.getEdgesAsync( scene, edges );
		signal?.throwIfAborted();

		if ( includeIntersectionEdges ) {

			if ( onProgress ) {

				onProgress( 0, 'Generating Intersection Edges' );

			}

			await edgeGenerator.getIntersectionEdgesAsync( scene, edges );
			signal?.throwIfAborted();

		}

		edges = edges.filter( e => ! isYProjectedLineDegenerate( e ) );

		if ( edges.length === 0 ) {

			return new ProjectionResult();

		}

		onProgress( 0, 'Projecting Edges' );

		//

		// allocate a buffer of edges for at most the requested capacity
		const batchCapacity = Math.min( batchSize, edges.length );
		const edgeBufferData = new Float32Array( batchCapacity * edgeStruct.getLength() );
		const edgeBufferDataU32 = new Uint32Array( edgeBufferData.buffer );
		const edgeBufferAttribute = new StorageBufferAttribute( edgeBufferData, edgeStruct.getLength() );

		// overlap output buffer and atomic counter
		const overlapsAttribute = new IndirectStorageBufferAttribute( MAX_OVERLAPS_COUNT, overlapRecordStruct.getLength(), Uint32Array );
		const bufferPointersAttribute = new IndirectStorageBufferAttribute( 1, 1 );
		const overflowFlagAttribute = new IndirectStorageBufferAttribute( 1, 1 );

		const overlapsStorage = storage( overlapsAttribute, overlapRecordStruct ).setName( 'overlaps' );
		const bufferPointersStorage = storage( bufferPointersAttribute, 'uint' ).toAtomic();
		const overflowFlagStorage = storage( overflowFlagAttribute, 'uint' ).setName( 'overflowFlag' ).toAtomic();

		//

		// set up scene data
		const bvhComputeData = new ProjectionGeneratorBVHComputeData( meshes );
		bvhComputeData.update();
		bvhComputeData.fns.collectEdgeOverlaps = bvhComputeData.getCollectEdgeOverlapsFn( {
			overlapsStorage: overlapsStorage,
			bufferPointersStorage: bufferPointersStorage,
			overflowFlagStorage: overflowFlagStorage,
		} );

		// initialize kernels
		const edgeOverlapsKernel = new EdgeOverlapsKernel();
		edgeOverlapsKernel.setWorkgroupSize( 64, 1, 1 );
		edgeOverlapsKernel.edges = edgeBufferAttribute;
		edgeOverlapsKernel.bvhData = bvhComputeData;

		const zeroOutKernel = new ZeroOutBufferKernel();
		zeroOutKernel.setWorkgroupSize( 1, 1, 1 );

		//
		const intervalsByEdge = new Map();
		let progress = 0;
		const promises = [];
		const edgeStructStride = edgeStruct.getLength();

		const jobQueue = new JobQueue();
		const onAbort = () => jobQueue.cancelAll();
		signal?.addEventListener( 'abort', onAbort );

		const runJob = async ( start, count ) => {

			if ( signal?.aborted ) {

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
			edgeOverlapsKernel.edgesToProcess = count;
			renderer.compute( edgeOverlapsKernel.kernel, edgeOverlapsKernel.getDispatchSize( count ) );

			const [ overlaps, bufferPointers, overflowBuffer ] = await Promise.all( [
				renderer.getArrayBufferAsync( overlapsAttribute ),
				renderer.getArrayBufferAsync( bufferPointersAttribute ),
				renderer.getArrayBufferAsync( overflowFlagAttribute ),
			] );

			if ( signal?.aborted ) {

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

				onProgress( progress / edges.length, 'Projecting Edges' );

			}

		};

		// enqueue initial jobs
		for ( let e = 0; e < edges.length; e += batchCapacity ) {

			promises.push( jobQueue.add( runJob, [ e, Math.min( batchCapacity, edges.length - e ) ] ) );

		}

		// drain — sequential iteration naturally picks up overflow sub-jobs added to promises
		try {

			for ( let i = 0; i < promises.length; i ++ ) {

				await promises[ i ];

			}

		} finally {

			signal?.removeEventListener( 'abort', onAbort );
			// overlapsAttribute.dispose();
			// bufferPointersAttribute.dispose();
			// overflowFlagAttribute.dispose();
			// edgeBufferAttribute.dispose();

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
