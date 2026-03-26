import { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import { storage, uniform } from 'three/tsl';
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
		const edgeStorage = storage( edgeBufferAttribute, edgeStruct ).toReadOnly().setName( 'edges' );

		// store the triangle / edge pairs to
		const triEdgeAttribute = new IndirectStorageBufferAttribute( batchCapacity * 64, triEdgePairStruct.getLength() );
		const triEdgeStorage = storage( triEdgeAttribute, triEdgePairStruct ).setName( 'TriEdge' );

		const triEdgeSizeAttribute = new IndirectStorageBufferAttribute( 3, 1 );
		const triEdgeSizeStorage = storage( triEdgeSizeAttribute, 'uint' );

		const overlapsAttribute = new IndirectStorageBufferAttribute( batchCapacity * 64, overlapRecordStruct.getLength() );
		const overlapStorage = storage( overlapsAttribute, overlapRecordStruct ).setName( 'overlaps' );

		const overlapsSizeAttribute = new IndirectStorageBufferAttribute( 3, 1 );
		const overlapSizeStorage = storage( overlapsSizeAttribute, 'uint' );

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

		// initialize kernels
		const edgePairsKernel = new EdgePairsKernel();
		edgePairsKernel.edges = edgeBufferAttribute;
		edgePairsKernel.bvhData = bvhComputeData;

		const edgeOverlapsKernel = new EdgeOverlapsKernel();
		edgeOverlapsKernel.pairs = triEdgeAttribute;
		edgeOverlapsKernel.pairsSize = triEdgeSizeAttribute;
		edgeOverlapsKernel.bvhData = bvhComputeData;

		//

		// accumulate potential triangle-edge overlap pairs
		renderer.compute( edgePairsKernel.kernel, edgePairsKernel.getDispatchSize( edgeBatchCapacity ) );

		// generate all overlaps
		renderer.compute( edgeOverlapsKernel.kernel, edgeOverlapsKernel.getDispatchSize( edgeBatchCapacity * 64 ) );

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

		return collector;

	}

}
