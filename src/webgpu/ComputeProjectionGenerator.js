import { getAllMeshes } from '../utils/getAllMeshes.js';
import { EdgeGenerator } from '../EdgeGenerator.js';
import { isYProjectedLineDegenerate } from '../utils/triangleLineUtils.js';

export class ComputeProjectionGenerator {

	constructor( renderer ) {

		this.renderer = renderer;
		this.angleThreshold = 50;
		this.includeIntersectionEdges = true;
		this.clipY = null;

	}

	async generate( scene ) {

		const { angleThreshold, includeIntersectionEdges, clipY, renderer } = this;

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

		// 3. construct an object bvh from the meshes

		// 4. pack the meshes into a compute buffer using the in-development BVHComputeData class
		// for three-mesh-bvh

		// CPU loop - advances a head pointer through the edge list until all edges are processed:
		//
		// 		5. pack the next batch of edges (from head pointer, up to edge storage buffer capacity)
		//         into a storage buffer for upload to the GPU
		//
		// 		6. run a dry-run kernel (one thread per edge) to count the number of overlaps each edge
		//         will produce via BVH traversal. read back the counts and determine how many edges fit
		//         within the overlap output buffer capacity. advance the head pointer by this amount.
		//
		// 		7. run the main kernel for the fitting edges, using BVH traversal to find overlapping
		//         triangles and writing overlap intervals to an atomic output buffer
		//
		// 		8. read back the overlap buffer and count. sort and merge overlaps per edge on the CPU
		//         and push the resulting visible line segments onto the output array.
		//
		// 		9. reset the output buffer atomic counter and continue from the new head pointer position

		// 9. generate the final output

	}

}
