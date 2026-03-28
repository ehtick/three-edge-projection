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
import { ProjectionResult } from '../ProjectionGenerator.js';

// TODO: edge splitting — long edges that span a large portion of the scene create heavy single-thread
// work in kernel 2. splitting them into sub-edges at pack time distributes that work across more
// threads. each sub-edge would carry tStart/tEnd (its [0,1] range within the original edge) so the
// GPU overlap t0/t1 values can be remapped back to original-edge space on the CPU before merging.

// TODO: Consider storing the ranges with multiple edges clipped per thread to reduce the array size needed

// TODO: initialize the compute kernels + buffers once to avoid construction overhead

// TODO: expose method for gathering edges per mesh

// TODO: handle edges iteratively

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

		// collect meshes
		const meshes = getAllMeshes( scene );

		// generate edges
		const edgeGenerator = new EdgeGenerator();
		edgeGenerator.thresholdAngle = angleThreshold;
		edgeGenerator.clipY = clipY;

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
		const batchCapacity = Math.min( edgeBatchCapacity, edges.length );
		const edgeBufferData = new Float32Array( batchCapacity * edgeStruct.getLength() );
		const edgeBufferDataU32 = new Uint32Array( edgeBufferData.buffer );
		const edgeBufferAttribute = new StorageBufferAttribute( edgeBufferData, edgeStruct.getLength() );

		// store the triangle / edge pairs to
		const triEdgePairsAttribute = new IndirectStorageBufferAttribute( batchCapacity * 70, triEdgePairStruct.getLength() );
		const triEdgePairsSizeAttribute = new IndirectStorageBufferAttribute( 3, 1 );
		const overlapsAttribute = new IndirectStorageBufferAttribute( batchCapacity * 70, overlapRecordStruct.getLength() );
		const overlapsSizeAttribute = new IndirectStorageBufferAttribute( 3, 1 );
		const overflowFlagAttribute = new IndirectStorageBufferAttribute( 1, 1 );

		const triEdgePairsStorage = storage( triEdgePairsAttribute, triEdgePairStruct ).setName( 'TriEdge' );
		const triEdgePairsSizeStorage = storage( triEdgePairsSizeAttribute, 'uint' ).toAtomic();
		const overflowFlagStorage = storage( overflowFlagAttribute, 'uint' ).setName( 'overflowFlag' ).toAtomic();

		// fill out the edges array
		const edgeStructStride = edgeStruct.getLength();
		for ( let i = 0; i < batchCapacity; i ++ ) {

			const { start, end } = edges[ i ];
			const offset = i * edgeStructStride;
			start.toArray( edgeBufferData, offset );
			end.toArray( edgeBufferData, offset + 3 );
			edgeBufferDataU32[ offset + 6 ] = i;

		}

		edgeBufferAttribute.needsUpdate = true;

		//

		// set up scene data
		const bvhComputeData = new ProjectionGeneratorBVHComputeData( meshes );
		bvhComputeData.update();
		bvhComputeData.fns.collectTriEdgePairs = bvhComputeData.getCollectTriEdgePairsFn( {
			pairsStorage: triEdgePairsStorage,
			pairCountsStorage: triEdgePairsSizeStorage,
			overflowFlagStorage: overflowFlagStorage,
		} );

		// initialize kernels
		const edgePairsKernel = new EdgePairsKernel();
		edgePairsKernel.edges = edgeBufferAttribute;
		edgePairsKernel.bvhData = bvhComputeData;

		const edgeOverlapsKernel = new EdgeOverlapsKernel();
		edgeOverlapsKernel.pairs = triEdgePairsAttribute;
		edgeOverlapsKernel.pairsSize = triEdgePairsSizeAttribute;
		edgeOverlapsKernel.bvhData = bvhComputeData;
		edgeOverlapsKernel.edges = edgeBufferAttribute;
		edgeOverlapsKernel.overlaps = overlapsAttribute;
		edgeOverlapsKernel.overlapsSize = overlapsSizeAttribute;

		//

		// accumulate potential triangle-edge overlap pairs
		renderer.compute( edgePairsKernel.kernel, edgePairsKernel.getDispatchSize( batchCapacity ) );

		// read back actual pair count before dispatching K3
		// const pairCountBuf = await renderer.getArrayBufferAsync( triEdgePairsSizeAttribute );
		// const pairCount = new Uint32Array( pairCountBuf )[ 1 ];

		// generate all overlaps — dispatch only over valid pairs
		renderer.compute( edgeOverlapsKernel.kernel, edgeOverlapsKernel.getDispatchSize( 3964384 ) );

		//

		// read result data back
		const [ overlaps, overlapsSize ] = await Promise.all( [
			renderer.getArrayBufferAsync( overlapsAttribute ),
			renderer.getArrayBufferAsync( overlapsSizeAttribute ),
		] );


		const overlapsF32 = new Float32Array( overlaps );
		const overlapsU32 = new Uint32Array( overlaps );
		const overlapsSizeU32 = new Uint32Array( overlapsSize );
		const stride = overlapRecordStruct.getLength();
		const intervalsByEdge = new Map();

		for ( let i = 0; i < overlapsSizeU32[ 0 ]; i ++ ) {

			const index = i * stride;
			const ei = overlapsU32[ index + 0 ];
			const t0 = overlapsF32[ index + 1 ];
			const t1 = overlapsF32[ index + 2 ];

			if ( ! intervalsByEdge.has( ei ) ) {

				intervalsByEdge.set( ei, [] );

			}

			intervalsByEdge.get( ei ).push( [ t0, t1 ] );

		}

		// sort, merge, and convert each edge's hidden intervals to visible/hidden line segments.
		// edges absent from intervalsByEdge had no occluding triangles and are fully visible.
		const collector = new ProjectionResult();
		for ( let i = 0; i < batchCapacity; i ++ ) {

			const edgeIndex = i;
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

		const overflow = new Uint32Array( await renderer.getArrayBufferAsync( overflowFlagAttribute ) );
		console.log( overflow[ 0 ] );

		return collector;

	}

}
