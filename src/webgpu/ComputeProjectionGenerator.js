import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';
import { getAllMeshes } from '../utils/getAllMeshes.js';
import { EdgeGenerator } from '../EdgeGenerator.js';
import { isYProjectedLineDegenerate } from '../utils/triangleLineUtils.js';
import { ProjectionGeneratorBVHComputeData } from './ProjectionGeneratorBVHComputeData.js';
import { edgeStruct } from './nodes/structs.wgsl.js';

export class ComputeProjectionGenerator {

	constructor( renderer ) {

		this.renderer = renderer;
		this.angleThreshold = 50;
		this.includeIntersectionEdges = true;
		this.clipY = null;
		this.edgeBatchCapacity = 1000000;

	}

	async generate( scene ) {

		const { angleThreshold, includeIntersectionEdges, clipY, edgeBatchCapacity } = this;

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
		const edgeStorage = storage( edgeBufferAttribute, edgeStruct ).toReadOnly().setName( 'edges' ); // eslint-disable-line no-unused-vars

		// CPU loop - advances a head pointer through the edge list until all edges are processed:
		const edgeStructStride = edgeStruct.getLength();
		const result = [];
		let headIndex = 0;
		while ( headIndex < edges.length ) {

			const batchSize = Math.min( batchCapacity, edges.length - headIndex );

			// 4. pack the next batch of edges into the storage buffer
			for ( let i = 0; i < batchSize; i ++ ) {

				const edgeIndex = headIndex + i;
				const { start, end } = edges[ edgeIndex ];
				const offset = i * edgeStructStride;
				start.toArray( edgeBufferData, offset );
				end.toArray( edgeBufferData, offset + 3 );
				edgeBufferDataU32[ offset + 6 ] = edgeIndex;

			}

			edgeBufferAttribute.needsUpdate = true;

			// 5. run a dry-run kernel (one thread per edge) to count the number of overlaps each edge
			//    will produce via BVH traversal. read back the counts and determine how many edges fit
			//    within the overlap output buffer capacity. advance the head pointer by this amount.

			// 6. run the main kernel for the fitting edges, using BVH traversal to find overlapping
			//    triangles and writing overlap intervals to an atomic output buffer

			// 7. read back the overlap buffer and count. sort and merge overlaps per edge on the CPU
			//    and push the resulting visible line segments onto the output array.

			// 8. reset the output buffer atomic counter and continue from the new head pointer position
			headIndex += batchSize;

		}

		// 9. generate the final output
		return result;

	}

}
