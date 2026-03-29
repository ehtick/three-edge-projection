import { globalId, storage } from 'three/tsl';
import { wgslTagFn } from '../lib/nodes/WGSLTagFnNode.js';
import { ComputeKernel } from '../utils/ComputeKernel.js';
import { proxy } from '../lib/nodes/NodeProxy.js';
import { StorageBufferAttribute } from 'three/webgpu';
import { edgeStruct, overlapRecordStruct, triEdgePairStruct } from '../nodes/structs.wgsl.js';
import { LineWGSL, TriWGSL } from '../nodes/primitives.js';
import { getProjectedOverlapRange, isLineTriangleEdge, trimToBeneathTriPlane } from '../nodes/overlapFunctions.wgsl.js';
import { constants } from '../nodes/common.wgsl.js';

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
			pairsSize: storage( new StorageBufferAttribute( 1, 1, Uint32Array ), 'uint' ).toReadOnly().setName( 'triEdgesSize' ),
			edges: storage( new StorageBufferAttribute( 1, edgeStruct.getLength(), Uint32Array ), edgeStruct ).toReadOnly(),
			overlapsSize: storage( new StorageBufferAttribute( 1, 3, Uint32Array ), 'uint' ).toAtomic(),
			overlaps: storage( new StorageBufferAttribute( 1, overlapRecordStruct.getLength(), Uint32Array ), overlapRecordStruct ),
		};

		const indexStorage = proxy( 'bvhData.value.storage.index', params );
		const attributesStorage = proxy( 'bvhData.value.storage.attributes', params );
		const transformsStorage = proxy( 'bvhData.value.storage.transforms', params );

		const computeTriangleEdgeOverlapFn = wgslTagFn/* wgsl */`
			fn computeTriangleEdgeOverlap( edgeIndex: u32, objectIndex: u32, triIndex: u32 ) -> void {

				// tri
				var tri: ${ TriWGSL.struct };
				let i0 = ${ indexStorage }[ triIndex * 3u + 0u ];
				let i1 = ${ indexStorage }[ triIndex * 3u + 1u ];
				let i2 = ${ indexStorage }[ triIndex * 3u + 2u ];

				let matrixWorld = ${ transformsStorage }[ objectIndex ].matrixWorld;
				tri.a = ( matrixWorld * vec4f( ${ attributesStorage }[ i0 ].position.xyz, 1.0 ) ).xyz;
				tri.b = ( matrixWorld * vec4f( ${ attributesStorage }[ i1 ].position.xyz, 1.0 ) ).xyz;
				tri.c = ( matrixWorld * vec4f( ${ attributesStorage }[ i2 ].position.xyz, 1.0 ) ).xyz;

				// back-face cull based on per-object side (0=double, 1=front, -1=back)
				let inverted  = determinant( matrixWorld ) < 0.0;
				let triNormal = cross( tri.b - tri.a, tri.c - tri.a );
				let side = ${ transformsStorage }[ objectIndex ].side;
				if ( side != ${ constants.DOUBLE_SIDE } ) {

					let faceUp = ( triNormal.y > 0.0 ) != inverted;
					if ( faceUp == ( side == ${ constants.BACK_SIDE } ) ) {

						return;

					}

				}

				let triMaxY = max( max( tri.a.y, tri.b.y ), tri.c.y );
				let triMinY = min( min( tri.a.y, tri.b.y ), tri.c.y );

				// line
				var line: ${ LineWGSL.struct };
				line.start = vec3f(
					${ params.edges }[ edgeIndex ].start[ 0 ],
					${ params.edges }[ edgeIndex ].start[ 1 ],
					${ params.edges }[ edgeIndex ].start[ 2 ]
				);
				line.end = vec3f(
					${ params.edges }[ edgeIndex ].end[ 0 ],
					${ params.edges }[ edgeIndex ].end[ 1 ],
					${ params.edges }[ edgeIndex ].end[ 2 ]
				);

				let lineMinY = min( line.start.y, line.end.y );
				let lineMaxY = max( line.start.y, line.end.y );

				// skip triangles entirely below the edge
				if ( triMaxY <= lineMinY ) {

					return;

				}

				// skip if the edge lies on this triangle
				if ( ${ isLineTriangleEdge }( tri, line ) ) {

					return;

				}

				// trim edge to the portion below the triangle plane; if the
				// entire line is already below the triangle, use the full line
				var beneathLine: ${ LineWGSL.struct };
				if ( lineMaxY < triMinY ) {

					beneathLine = line;

				} else if ( ! ${ trimToBeneathTriPlane }( tri, line, &beneathLine ) ) {

					return;

				}

				// skip degenerate trimmed segments
				// TODO: add a "distant" utility function
				if ( length( beneathLine.end - beneathLine.start ) < ${ constants.DIST_THRESHOLD } ) {

					return;

				}

				// get projected overlap range in trimmed-edge space
				var overlapLine: ${ LineWGSL.struct };
				if ( ${ getProjectedOverlapRange }( beneathLine, tri, &overlapLine ) ) {

					// overlap line end points relative to "start"
					let dir = line.end - line.start;
					let v0 = overlapLine.start - line.start;
					let v1 = overlapLine.end - line.start;

					// get the [0, 1] t values
					let l = length( dir );
					var t0 = length( v0 ) / l;
					var t1 = length( v1 ) / l;

					t0 = min( max( t0, 0.0 ), 1.0 );
					t1 = min( max( t1, 0.0 ), 1.0 );

					if ( abs( t0 - t1 ) <= ${ constants.DIST_THRESHOLD } ) {

						return;

					}

					// claim a slot and write the overlap record
					let slot = atomicAdd( &${ params.overlapsSize }[ 0 ], 1u );
					if ( slot < arrayLength( &${ params.overlaps } ) ) {

						${ params.overlaps }[ slot ].edgeIndex = edgeIndex;
						${ params.overlaps }[ slot ].t0 = t0;
						${ params.overlaps }[ slot ].t1 = t1;

					}

				}

			}
		`;

		const shader = wgslTagFn/* wgsl */`
			fn compute( globalId: vec3u ) -> void {

				// pairsSize includes the total amount of pairs written to the
				// original buffer
				let pairIndex = atomicAdd( &${ params.overlapsSize }[ 1 ], 1u );
				if ( pairIndex >= ${ params.pairsSize }[ 1 ] ) {

					return;

				}

				let pair = ${ params.pairs }[ pairIndex ];
				${ computeTriangleEdgeOverlapFn }( pair.edgeIndex, pair.objectIndex, pair.triIndex );

			}
		`;

		super( shader( params ) );
		this.defineUniformAccessors( params );
		this.setWorkgroupSize( 64, 1, 1 );

	}

}
