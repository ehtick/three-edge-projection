import { uniform, globalId } from 'three/tsl';
import { wgslTagFn } from './lib/nodes/WGSLTagFnNode.js';
import { ComputeKernel } from './utils/ComputeKernel.js';

// Kernel 3 — one thread per (edge, triangle) pair.
// Reads each pre-computed pair record from the pairs buffer, fetches the
// corresponding edge endpoints and triangle vertices, runs the single-triangle
// overlap computation, and writes any resulting [t0, t1] interval to the
// overlaps buffer via atomic slot claiming.
//
// overlapsFn is obtained from ProjectionGeneratorBVHComputeData.getOverlapsFn()
// and has all buffer nodes captured inside it.
export class EdgeOverlapsKernel extends ComputeKernel {

	constructor( overlapsFn ) {

		const params = {
			globalId: globalId,
			pairCount: uniform( 0 ),
		};

		const shader = wgslTagFn/* wgsl */`
			fn compute(
				globalId: vec3u,
				pairCount: u32,
			) -> void {

				let pairIndex = globalId.x;
				if ( pairIndex >= pairCount ) { return; }

				${ overlapsFn }( pairIndex );

			}
		`;

		super( shader( params ) );
		this.defineUniformAccessors( params );

	}

}
