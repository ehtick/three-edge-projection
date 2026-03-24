import { uniform, globalId } from 'three/tsl';
import { wgslTagFn } from './lib/nodes/WGSLTagFnNode.js';
import { ComputeKernel } from './utils/ComputeKernel.js';

// Kernel 2 — one thread per edge.
// Traverses the BVH for each edge and writes qualifying (edgeIndex, objectIndex, triIndex)
// records to the pairs buffer using atomic slot claiming.  If the buffer overflows, the
// lowest overflowing edgeIndex is recorded and the caller can retry the remaining edges
// by re-running kernels 2 and 3 from that point.
//
// traversalFn is obtained from ProjectionGeneratorBVHComputeData.getTraversalFn() and
// already has the pairs buffer, pairCountNode, pairsCapacityNode, and overflowEdgeIndexNode
// captured inside it; passing writePairs=1 activates the write path.
export class EdgePairsKernel extends ComputeKernel {

	constructor( traversalFn, edgesNode ) {

		const params = {
			globalId: globalId,
			edgeCount: uniform( 0 ),
		};

		const shader = wgslTagFn/* wgsl */`
			fn compute(
				globalId: vec3u,
				edgeCount: u32,
			) -> void {

				let edgeIndex = globalId.x;
				if ( edgeIndex >= edgeCount ) { return; }

				let edgeStart = vec3f(
					${ edgesNode }[ edgeIndex ].start[ 0 ],
					${ edgesNode }[ edgeIndex ].start[ 1 ],
					${ edgesNode }[ edgeIndex ].start[ 2 ]
				);
				let edgeEnd = vec3f(
					${ edgesNode }[ edgeIndex ].end[ 0 ],
					${ edgesNode }[ edgeIndex ].end[ 1 ],
					${ edgesNode }[ edgeIndex ].end[ 2 ]
				);

				${ traversalFn }( edgeIndex, edgeStart, edgeEnd, 1u );

			}
		`;

		super( shader( params ) );
		this.defineUniformAccessors( params );

	}

}
