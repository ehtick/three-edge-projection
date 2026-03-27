import { globalId, storage } from 'three/tsl';
import { wgslTagFn } from '../lib/nodes/WGSLTagFnNode.js';
import { ComputeKernel } from '../utils/ComputeKernel.js';
import { proxyFn } from '../lib/nodes/NodeProxy.js';
import { StorageBufferAttribute } from 'three/webgpu';
import { edgeStruct } from '../nodes/structs.wgsl.js';

// kernel for gathering pairs of triangles / edges that have a potential overlap
export class EdgePairsKernel extends ComputeKernel {

	constructor() {

		const params = {
			bvhData: { value: null },
			globalId: globalId,
			edges: storage( new StorageBufferAttribute( 1, 1, Uint32Array ), edgeStruct ).toReadOnly().setName( 'edges' ),
		};

		const edges = params.edges;
		const traversalFn = proxyFn( 'bvhData.value.fns.collectTriEdgePairs', params );
		const shader = wgslTagFn/* wgsl */`
			fn compute( globalId: vec3u ) -> void {

				let edgeIndex = globalId.x;
				let edgeListLength = arrayLength( &${ edges } );
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

				${ traversalFn }( edgeIndex, edgeStart, edgeEnd );

			}
		`;

		super( shader( params ) );
		this.defineUniformAccessors( params );
		this.setWorkgroupSize( 64, 1, 1 );

	}

}
