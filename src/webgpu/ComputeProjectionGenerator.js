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

// TODO: expose method for gathering edges per mesh

const OVERLAPS_PER_EDGE = 100;
export class ComputeProjectionGenerator {

	constructor( renderer ) {

		this.renderer = renderer;
		this.angleThreshold = 50;
		this.batchSize = 50000;
		this.includeIntersectionEdges = true;
		this.clipY = null;

	}

	async generate( scene ) {

		const { renderer, angleThreshold, includeIntersectionEdges, clipY, batchSize } = this;

		// collect meshes
		const meshes = getAllMeshes( scene );

		// generate edges
		const edgeGenerator = new EdgeGenerator();
		edgeGenerator.thresholdAngle = angleThreshold;
		edgeGenerator.clipY = clipY;

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
		const triEdgePairsAttribute = new IndirectStorageBufferAttribute( batchCapacity * OVERLAPS_PER_EDGE, triEdgePairStruct.getLength() );
		const triEdgePairsCountAttribute = new IndirectStorageBufferAttribute( 2, 1 );
		const overlapsAttribute = new IndirectStorageBufferAttribute( batchCapacity * OVERLAPS_PER_EDGE, overlapRecordStruct.getLength() );
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
		edgePairsKernel.edges = edgeBufferAttribute;
		edgePairsKernel.bvhData = bvhComputeData;

		const edgeOverlapsKernel = new EdgeOverlapsKernel();
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
		let batchStep = batchCapacity;
		for ( let e = 0; e < edges.length; e += batchStep ) {

			const iterationCount = Math.min( batchStep, edges.length - e );

			// fill out the edges array
			const edgeStructStride = edgeStruct.getLength();
			for ( let i = 0; i < iterationCount; i ++ ) {

				const { start, end } = edges[ e + i ];
				const offset = i * edgeStructStride;
				start.toArray( edgeBufferData, offset );
				end.toArray( edgeBufferData, offset + 3 );
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
			edgePairsKernel.edgesToProcess = iterationCount;
			renderer.compute( edgePairsKernel.kernel, edgePairsKernel.getDispatchSize( iterationCount ) );

			const overflow = new Uint32Array( await renderer.getArrayBufferAsync( overflowFlagAttribute ) );
			if ( overflow > 0 ) {

				batchStep = Math.ceil( batchStep * 0.5 );
				e -= batchStep;
				continue;

			}

			// read back actual pair count before dispatching
			const pairCountBuf = await renderer.getArrayBufferAsync( triEdgePairsCountAttribute );
			const pairCount = new Uint32Array( pairCountBuf )[ 1 ];

			const dispatchSize = edgeOverlapsKernel.getDispatchSize( pairCount )[ 0 ];
			const dispatchStepSize = Math.min( dispatchSize, 65535 );

			for ( let i = 0; i < dispatchSize; i += dispatchStepSize ) {

				// generate all overlaps — dispatch only over valid pairs
				renderer.compute( zeroOutKernel.kernel, [ 1, 1, 1 ] );
				renderer.compute( edgeOverlapsKernel.kernel, [ dispatchStepSize, 1, 1 ] );

				// read result data back
				const [ overlaps, bufferPointers ] = await Promise.all( [
					renderer.getArrayBufferAsync( overlapsAttribute ),
					renderer.getArrayBufferAsync( bufferPointersAttribute ),
				] );

				const overlapsF32 = new Float32Array( overlaps );
				const overlapsU32 = new Uint32Array( overlaps );
				const bufferPointersU32 = new Uint32Array( bufferPointers );
				const stride = overlapRecordStruct.getLength();

				for ( let oi = 0, ol = bufferPointersU32[ 0 ]; oi < ol; oi ++ ) {

					const index = oi * stride;
					const ei = e + overlapsU32[ index + 0 ];
					const t0 = overlapsF32[ index + 1 ];
					const t1 = overlapsF32[ index + 2 ];

					if ( ! intervalsByEdge.has( ei ) ) {

						intervalsByEdge.set( ei, [] );

					}

					insertOverlap( [ t0, t1 ], intervalsByEdge.get( ei ) );

				}

			}

			// try to grow the batch step towards the original stride
			if ( batchStep < batchCapacity ) {

				e += batchStep;
				batchStep = Math.min( batchStep * 2.0, batchCapacity );
				e -= batchStep;

			}

		}

		// convert each edge's hidden intervals to visible/hidden line segments.
		// edges absent from intervalsByEdge had no occluding triangles and are fully visible.
		const collector = new ProjectionResult();
		for ( let i = 0; i < edges.length; i ++ ) {

			const intervals = intervalsByEdge.get( i ) || [];
			overlapsToLines( edges[ i ], intervals, false, collector.visibleEdges );
			overlapsToLines( edges[ i ], intervals, true, collector.hiddenEdges );

		}

		return collector;

	}

}
