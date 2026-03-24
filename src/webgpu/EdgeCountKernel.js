import { uniform, globalId } from 'three/tsl';
import { wgslTagFn } from './lib/nodes/WGSLTagFnNode.js';
import { ComputeKernel } from './utils/ComputeKernel.js';

// Kernel 1 — one thread per edge.
// Traverses the BVH for each edge and atomically accumulates the total number of
// qualifying (edge, triangle) pairs into pairCountNode[0].  The result is used by
// the caller to size the pairs buffer and, optionally, to populate an indirect
// dispatch buffer for kernel 3.
//
// traversalFn is obtained from ProjectionGeneratorBVHComputeData.getTraversalFn() and
// already has the pairs buffer nodes captured inside it; passing writePairs=0 runs
// the count-only path with no writes to the pairs buffer.
// pairCountNode must be bound as array<atomic<u32>> (read_write storage) and
// pre-cleared to 0 before dispatch.
export class EdgeCountKernel extends ComputeKernel {

	constructor( traversalFn, edgesNode, pairCountNode ) {

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

				let count = ${ traversalFn }( edgeIndex, edgeStart, edgeEnd, 0u );
				atomicAdd( &${ pairCountNode }[ 0 ], count );

			}
		`;

		super( shader( params ) );
		this.defineUniformAccessors( params );

	}

}
