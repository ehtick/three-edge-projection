import { globalId, storage } from 'three/tsl';
import { wgslTagFn } from '../lib/nodes/WGSLTagFnNode.js';
import { ComputeKernel } from '../utils/ComputeKernel.js';
import { proxyFn } from '../lib/nodes/NodeProxy.js';
import { StorageBufferAttribute } from 'three/webgpu';
import { triEdgePairStruct } from '../nodes/structs.wgsl.js';

// Kernel 3 — one thread per (edge, triangle) pair.
// Reads each pre-computed pair record from the pairs buffer, fetches the
// corresponding edge endpoints and triangle vertices, runs the single-triangle
// overlap computation, and writes any resulting [t0, t1] interval to the
// overlaps buffer via atomic slot claiming.
//
// overlapsFn is obtained from ProjectionGeneratorBVHComputeData.getOverlapsFn()
// and has all buffer nodes captured inside it.
export class EdgeOverlapsKernel extends ComputeKernel {

	constructor() {

		const params = {
			bvhData: { value: null },
			globalId: globalId,
			pairs: storage( new StorageBufferAttribute( 1, triEdgePairStruct.getLength(), Uint32Array ), triEdgePairStruct ).toReadOnly().setName( 'triEdges' ),
			pairsSize: storage( new StorageBufferAttribute( 1, 1, Uint32Array ), triEdgePairStruct ).toReadOnly().setName( 'triEdgesSize' ),
		};

		const overlapsFn = proxyFn( params, 'bvhData.fns.collectTriEdgeOverlaps' );

		const shader = wgslTagFn/* wgsl */`
			fn compute( globalId: vec3u ) -> void {

				// pairsSize includes the total amount of pairs written to the
				// original buffer
				let pairIndex = globalId.x;
				if ( pairIndex >= ${ params.pairsSize }[ 1 ] ) {

					return;

				}

				let pair = ${ params.pairs }[ pairIndex ];
				${ overlapsFn }( pair.edgeIndex, pair.objectIndex, pair.triIndex );

			}
		`;

		super( shader( params ) );
		this.defineUniformAccessors( params );
		this.setWorkgroupSize( 64, 1, 1 );

	}

}
