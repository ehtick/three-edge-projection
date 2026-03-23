import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';
import { getAllMeshes } from '../utils/getAllMeshes.js';
import { EdgeGenerator } from '../EdgeGenerator.js';
import { isYProjectedLineDegenerate } from '../utils/triangleLineUtils.js';
import { ProjectionGeneratorBVHComputeData } from './ProjectionGeneratorBVHComputeData.js';
import { edgeStruct } from './nodes/structs.wgsl.js';

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

			// 5. kernel 1 (count): one thread per edge traverses the BVH and atomically accumulates
			//    the total number of triangle/edge pairs into a single counter. the result is written
			//    into an indirect dispatch buffer so kernel 2 can be dispatched without a CPU readback.
			//    advance the head pointer by batchSize.

			// 6. kernel 2 (pairs): one thread per edge traverses the BVH again and writes
			//    { edgeIndex, objectIndex, triIndex } records to a pairs buffer using atomicAdd to
			//    claim each slot. if the buffer is full the thread early-outs, leaving those edges
			//    unprocessed. the number of pairs written is stored for indirect dispatch of kernel 3.
			//    kernel 2 also writes the index of the first edge that did not fit, so a retry knows
			//    where to resume.

			// 7. kernel 3 (overlaps): one thread per pair, dispatched indirectly from kernel 2's count.
			//    reads the triangle/edge pair, transforms triangle vertices to world space, computes the
			//    projected overlap, and writes { edgeIndex, t0, t1 } to an overlaps buffer via atomicAdd.
			//    after kernel 3 completes, check the overflow flag from kernel 2. if overflow occurred,
			//    clear the pairs buffer counters and re-run kernels 2 and 3 for the remaining edges.

			// 8. read back the overlaps buffer and count. group intervals by edgeIndex, merge overlapping
			//    [t0, t1] ranges per edge on the CPU, convert to 3D line segments, and push onto result.

			// 9. reset the pairs and overlaps buffer atomic counters and continue from the new head pointer.
			headIndex += batchSize;

		}

		// 9. generate the final output
		return result;

	}

}
