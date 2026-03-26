import { globalId, storage } from 'three/tsl';
import { wgslTagFn } from '../lib/nodes/WGSLTagFnNode.js';
import { ComputeKernel } from '../utils/ComputeKernel.js';
import { proxyFn } from '../lib/nodes/NodeProxy.js';
import { StorageBufferAttribute } from 'three/webgpu';
import { edgeStruct } from '../nodes/structs.wgsl.js';

// Traverses the BVH for each edge and writes qualifying (edgeIndex, objectIndex, triIndex)
// records to the pairs buffer using atomic slot claiming.  If the buffer overflows, the
// lowest overflowing edgeIndex is recorded and the caller can retry the remaining edges
// by re-running kernels 2 and 3 from that point.
//
// traversalFn is obtained from ProjectionGeneratorBVHComputeData.getTraversalFn() and
// already has the pairs buffer, pairCountNode, pairsCapacityNode, and overflowEdgeIndexNode
// captured inside it; passing writePairs=1 activates the write path.
export class EdgePairsKernel extends ComputeKernel {

	constructor() {

		const params = {
			bvhData: { value: null },
			globalId: globalId,
			edges: storage( new StorageBufferAttribute( 1, 1, Uint32Array ), edgeStruct ).toReadOnly().setName( 'edges' ),
		};

		const edges = params.edges;
		const traversalFn = proxyFn( params, 'bvhData.fns.gatherTriangles' );
		const shader = wgslTagFn/* wgsl */`
			fn compute( globalId: vec3u ) -> void {

				let edgeIndex = globalId.x;
				let edgeListLength = arrayLength( ${ edges } );
				if ( edgeIndex >= edgeListLength ) {

					return;

				}

				let edgeStart = vec3f(
					${ edges }[ edgeIndex ].start[ 0 ],
					${ edges }[ edgeIndex ].start[ 1 ],
					${ edges }[ edgeIndex ].start[ 2 ]
				);
				let edgeEnd = vec3f(
					${ edges }[ edgeIndex ].end[ 0 ],
					${ edges }[ edgeIndex ].end[ 1 ],
					${ edges }[ edgeIndex ].end[ 2 ]
				);

				${ traversalFn }( edgeIndex, edgeStart, edgeEnd, 1u );

			}
		`;

		super( shader( params ) );
		this.defineUniformAccessors( params );
		this.setWorkgroupSize( 64, 1, 1 );

	}

}
