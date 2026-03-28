import { BackSide, BufferAttribute, BufferGeometry, DoubleSide, FrontSide, SkinnedMesh } from 'three';
import { StructTypeNode } from 'three/webgpu';
import { BVHComputeData } from './lib/BVHComputeData.js';
import { MeshBVH, SAH, SkinnedMeshBVH } from 'three-mesh-bvh';
import { wgslTagFn } from './lib/nodes/WGSLTagFnNode.js';
import { bvhNodeStruct, bvhNodeBoundsStruct } from './lib/wgsl/structs.wgsl.js';
import { transformBVHBounds } from './nodes/utils.wgsl.js';
import { constants as overlapConstants } from './nodes/common.wgsl.js';
import {
	trimToBeneathTriPlane,
	getProjectedOverlapRange,
	isLineTriangleEdge,
} from './nodes/overlapFunctions.wgsl.js';
import { LineWGSL, TriWGSL } from './nodes/primitives.js';

// Shape struct carrying world-space line endpoints plus the object-to-world
// matrix (set by transformShapeFn; identity at top level so world-space
// bounds pass through unchanged) and the transform buffer index.
const edgeLineShapeStruct = new StructTypeNode( {
	worldStart: 'vec3f',
	worldEnd: 'vec3f',
	matrixWorld: 'mat4x4f',
	objectIndex: 'uint',
	edgeIndex: 'uint',
}, 'EdgeLineShape' );

// Minimal result struct satisfying the shapecast interface.
// We never set didHit = true, so no best-hit pruning occurs.
const edgeOverlapResultStruct = new StructTypeNode( {
	didHit: 'bool',
	dist: 'float',
	objectIndex: 'uint',
}, 'EdgeOverlapResult' );

// Extended transform struct that adds a per-object "side" field for back-face
// culling. Layout matches the base struct; "side" reuses the _alignment0 slot.
// Values: 0 = DoubleSide (no cull), 1 = FrontSide, -1 = BackSide.
const projectionTransformStruct = new StructTypeNode( {
	matrixWorld: 'mat4x4f',
	inverseMatrixWorld: 'mat4x4f',
	nodeOffset: 'uint',
	visible: 'uint',
	side: 'int',
	_alignment0: 'uint',
}, 'ProjectionTransformStruct' );

// Projection-generator-specific BVHComputeData that only requires position
// attributes and auto-generates missing BVHs.
export class ProjectionGeneratorBVHComputeData extends BVHComputeData {

	constructor( bvh, options = {} ) {

		super( bvh, {
			attributes: { position: 'vec4f' },
			...options,
		} );

		this.bvhMap = new Map();
		this.structs.transform = projectionTransformStruct;
		this._sharedFns = null;
		this._fns = null;

	}

	writeTransformData( info, premultiplyMatrix, writeOffset, targetBuffer ) {

		super.writeTransformData( info, premultiplyMatrix, writeOffset, targetBuffer );

		const { object, root } = info;
		let material = object.material;
		if ( Array.isArray( material ) ) {

			material = material[ object.geometry.groups[ root ].materialIndex ];

		}

		let sideValue;
		switch ( material.side ) {

			case DoubleSide:
				sideValue = 0;
				break;
			case FrontSide:
				sideValue = 1;
				break;
			case BackSide:
				sideValue = - 1;
				break;

		}

		const transformBufferU32 = new Uint32Array( targetBuffer );
		transformBufferU32[ writeOffset * projectionTransformStruct.getLength() + 34 ] = sideValue;

	}

	update() {

		super.update();
		this.bvhMap.clear();
		this._sharedFns = null;
		this._fns = null;

	}

	// Returns a WGSL function — fn prefix_ComputeOverlap( pairIndex: u32 ) -> void — that reads
	// one (edge, triangle) pair record, fetches the edge endpoints and triangle vertices,
	// runs the single-triangle overlap computation, and atomically writes the resulting
	// [t0, t1] interval to the overlaps buffer if one exists.
	getTriangleEdgeOverlapsFn( { edgesStorage, overlapsStorage, overlapsCountStorage } ) {

		const { storage } = this;
		const { DIST_THRESHOLD, DOUBLE_SIDE, BACK_SIDE } = overlapConstants;

		return wgslTagFn/* wgsl */`
			fn computeTriangleEdgeOverlap( edgeIndex: u32, objectIndex: u32, triIndex: u32 ) -> void {

				// tri
				var tri: ${ TriWGSL.struct };
				let i0 = ${ storage.index }[ triIndex * 3u + 0u ];
				let i1 = ${ storage.index }[ triIndex * 3u + 1u ];
				let i2 = ${ storage.index }[ triIndex * 3u + 2u ];

				let matrixWorld = ${ storage.transforms }[ objectIndex ].matrixWorld;
				tri.a = ( matrixWorld * vec4f( ${ storage.attributes }[ i0 ].position.xyz, 1.0 ) ).xyz;
				tri.b = ( matrixWorld * vec4f( ${ storage.attributes }[ i1 ].position.xyz, 1.0 ) ).xyz;
				tri.c = ( matrixWorld * vec4f( ${ storage.attributes }[ i2 ].position.xyz, 1.0 ) ).xyz;

				// back-face cull based on per-object side (0=double, 1=front, -1=back)
				let inverted  = determinant( matrixWorld ) < 0.0;
				let triNormal = cross( tri.b - tri.a, tri.c - tri.a );
				let side = ${ storage.transforms }[ objectIndex ].side;
				if ( side != ${ DOUBLE_SIDE } ) {

					let faceUp = ( triNormal.y > 0.0 ) != inverted;
					if ( faceUp == ( side == ${ BACK_SIDE } ) ) {

						return;

					}

				}

				let triMaxY = max( max( tri.a.y, tri.b.y ), tri.c.y );
				let triMinY = min( min( tri.a.y, tri.b.y ), tri.c.y );

				// line
				var line: ${ LineWGSL.struct };
				line.start = vec3f(
					${ edgesStorage }[ edgeIndex ].start[ 0 ],
					${ edgesStorage }[ edgeIndex ].start[ 1 ],
					${ edgesStorage }[ edgeIndex ].start[ 2 ]
				);
				line.end = vec3f(
					${ edgesStorage }[ edgeIndex ].end[ 0 ],
					${ edgesStorage }[ edgeIndex ].end[ 1 ],
					${ edgesStorage }[ edgeIndex ].end[ 2 ]
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
				if ( length( beneathLine.end - beneathLine.start ) < ${ DIST_THRESHOLD } ) {

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

					if ( abs( t0 - t1 ) <= ${ DIST_THRESHOLD } ) {

						return;

					}

					// claim a slot and write the overlap record
					let slot = atomicAdd( &${ overlapsCountStorage }[ 0 ], 1u );
					if ( slot < arrayLength( &${ overlapsStorage } ) ) {

						${ overlapsStorage }[ slot ].edgeIndex = edgeIndex;
						${ overlapsStorage }[ slot ].t0 = t0;
						${ overlapsStorage }[ slot ].t1 = t1;

					}

				}

			}
		`;

	}

	// Returns a WGSL function — fn traverse( edgeIndex, lineStart, lineEnd ) -> void —
	// that traverses the BVH for one edge and writes qualifying { edgeIndex, objectIndex, triIndex }
	// records to the pairs buffer using atomic slot claiming.
	//
	// pairCountsStorage is a 2-element array<atomic<u32>>:
	//   [0] write offset — claimed unconditionally via atomicAdd
	//   [1] dispatch count — incremented only when the claimed slot is within capacity; equals
	//       the number of valid pair records written and is used as K3's dispatch bound
	//
	// overflowFlagStorage is a 1-element array<atomic<u32>> that accumulates the number of
	// pairs that could not be written due to buffer overflow.
	//
	// NOTE: pairCountsStorage must be bound as array<atomic<u32>> (read_write storage).
	getCollectTriEdgePairsFn( { pairsStorage, pairCountsStorage, overflowFlagStorage } ) {

		const { storage } = this;
		const { DOUBLE_SIDE, BACK_SIDE } = overlapConstants;

		const boundsOrderFn = wgslTagFn/* wgsl */`
			fn boundsOrder( shape: ${ edgeLineShapeStruct }, splitAxis: u32, node: ${ bvhNodeStruct } ) -> bool {

				return true;

			}
		`;

		const intersectsBoundsFn = wgslTagFn/* wgsl */`
			fn intersectsBounds( shape: ${ edgeLineShapeStruct }, bounds: ${ bvhNodeBoundsStruct } ) -> f32 {

				// Transform bounds to world space. At the top level the shape matrix
				// is identity, so world-space bounds pass through unchanged.
				let aabb = ${ transformBVHBounds }( bounds, shape.matrixWorld );

				// Y-cull: bounds entirely below the line
				if ( aabb.max[ 1 ] <= min( shape.worldStart.y, shape.worldEnd.y ) ) {

					return - 1.0;

				}


				// TODO: confirm this is correct
				// XZ cull against the line's world-space XZ extents
				let lineMinX = min( shape.worldStart.x, shape.worldEnd.x );
				let lineMaxX = max( shape.worldStart.x, shape.worldEnd.x );
				let lineMinZ = min( shape.worldStart.z, shape.worldEnd.z );
				let lineMaxZ = max( shape.worldStart.z, shape.worldEnd.z );
				if (
					aabb.max[ 0 ] < lineMinX || aabb.min[ 0 ] > lineMaxX ||
					aabb.max[ 2 ] < lineMinZ || aabb.min[ 2 ] > lineMaxZ
				) {

					return - 1.0;

				}

				return 0.0;

			}
		`;

		const transformShapeFn = wgslTagFn/* wgsl */`
			fn transformShape( shape: ${ edgeLineShapeStruct }, inverseMatrix: mat4x4f, objectIndex: u32 ) -> ${ edgeLineShapeStruct } {

				var localShape = shape;
				localShape.matrixWorld = ${ storage.transforms }[ objectIndex ].matrixWorld;
				localShape.objectIndex = objectIndex;
				return localShape;

			}
		`;

		const transformResultFn = wgslTagFn/* wgsl */`
			fn transformResult( result: ptr<function, ${ edgeOverlapResultStruct }>, objectIndex: u32 ) -> void {

			}
		`;

		const intersectRangeFn = wgslTagFn/* wgsl */`
			fn traverseRange( shape: ${ edgeLineShapeStruct }, offset: u32, count: u32, bestDist: f32 ) -> ${ edgeOverlapResultStruct } {

				var result: ${ edgeOverlapResultStruct };
				result.didHit = false;
				result.dist = bestDist;

				var tri: ${ TriWGSL.struct };
				var line: ${ LineWGSL.struct };
				line.start = shape.worldStart;
				line.end = shape.worldEnd;

				let lineMinY = min( line.start.y, line.end.y );
				let matrixWorld = shape.matrixWorld;
				let inverted = determinant( matrixWorld ) < 0.0;

				for ( var ti = offset; ti < offset + count; ti = ti + 1u ) {

					let i0 = ${ storage.index }[ ti * 3u + 0u ];
					let i1 = ${ storage.index }[ ti * 3u + 1u ];
					let i2 = ${ storage.index }[ ti * 3u + 2u ];

					let localA = ${ storage.attributes }[ i0 ].position.xyz;
					let localB = ${ storage.attributes }[ i1 ].position.xyz;
					let localC = ${ storage.attributes }[ i2 ].position.xyz;

					tri.a = ( matrixWorld * vec4f( localA, 1.0 ) ).xyz;
					tri.b = ( matrixWorld * vec4f( localB, 1.0 ) ).xyz;
					tri.c = ( matrixWorld * vec4f( localC, 1.0 ) ).xyz;

					// back-face cull based on per-object side (0=double, 1=front, -1=back)
					let triNormal = cross( tri.b - tri.a, tri.c - tri.a );
					let side = ${ storage.transforms }[ shape.objectIndex ].side;
					if ( side != ${ DOUBLE_SIDE } ) {

						let faceUp = ( triNormal.y > 0.0 ) != inverted;
						if ( faceUp == ( side == ${ BACK_SIDE } ) ) {

							continue;

						}

					}

					// skip triangles entirely below the edge
					let highestTriY = max( max( tri.a.y, tri.b.y ), tri.c.y );
					if ( highestTriY <= lineMinY ) {

						continue;

					}

					// skip if the edge lies on this triangle
					if ( ${ isLineTriangleEdge }( tri, line ) ) {

						continue;

					}

					// claim a slot and write the pair record when in write mode
					let slot = atomicAdd( &${ pairCountsStorage }[ 0 ], 1u );
					if ( slot < arrayLength( &${ pairsStorage } ) ) {

						${ pairsStorage }[ slot ].edgeIndex   = shape.edgeIndex;
						${ pairsStorage }[ slot ].objectIndex = shape.objectIndex;
						${ pairsStorage }[ slot ].triIndex    = ti;
						atomicAdd( &${ pairCountsStorage }[ 1 ], 1u );

					} else {

						atomicAdd( &${ overflowFlagStorage }[ 0 ], 1u );

					}

				}

				return result;

			}
		`;

		const traversalFn = this.getShapecastFn( {
			name: 'collectTriEdgePairs',
			shapeStruct: edgeLineShapeStruct,
			resultStruct: edgeOverlapResultStruct,
			boundsOrderFn,
			intersectsBoundsFn,
			intersectRangeFn,
			transformShapeFn,
			transformResultFn,
		} );

		return wgslTagFn/* wgsl */`
			fn traverse( edgeIndex: u32, lineStart: vec3f, lineEnd: vec3f ) -> void {

				var shape: ${ edgeLineShapeStruct };
				shape.worldStart = lineStart;
				shape.worldEnd = lineEnd;
				shape.matrixWorld = mat4x4f(
					1.0, 0.0, 0.0, 0.0,
					0.0, 1.0, 0.0, 0.0,
					0.0, 0.0, 1.0, 0.0,
					0.0, 0.0, 0.0, 1.0
				);
				shape.objectIndex = 0u;
				shape.edgeIndex = edgeIndex;
				${ traversalFn }( shape );

			}
		`;

	}

	getBVH( object, instanceId, rangeTarget ) {

		if ( ! object.geometry.boundsTree ) {

			object.geometry.boundsTree = new MeshBVH( object.geometry, { strategy: SAH, maxLeafSize: 1 } );

		}

		const { bvhMap } = this;
		const bvh = super.getBVH( object, instanceId, rangeTarget );
		if ( bvhMap.has( bvh ) ) {

			const data = bvhMap.get( bvh );
			Object.assign( rangeTarget, data.range );

			// make sure the mesh and bvh are updated if it's being reused across updates
			if ( bvh !== data.bvh && bvh instanceof SkinnedMeshBVH ) {

				const sourceMesh = bvh.mesh;
				const clonedMesh = data.bvh.mesh;
				clonedMesh.matrixWorld
					.copy( sourceMesh.matrixWorld )
					.decompose( clonedMesh.position, clonedMesh.quaternion, clonedMesh.scale );

				bvh.refit();

			}

			return data.bvh;

		} else if ( bvh.indirect ) {

			// indirect bvhs are not supported since they cannot be unpacked coherently
			const proxyGeometry = new BufferGeometry();
			proxyGeometry.attributes = bvh.geometry.attributes;

			let array;
			if ( bvh.geometry.index ) {

				array = bvh.geometry.index.array.slice( rangeTarget.start, rangeTarget.count + rangeTarget.start );

			} else {

				const { start, count } = rangeTarget;
				array = new Uint32Array( count );
				for ( let i = 0, l = rangeTarget.count; i < l; i ++ ) {

					array[ i ] = start + i;

				}

			}

			proxyGeometry.index = new BufferAttribute( array, 1 );
			rangeTarget.start = 0;

			let newBVH;
			if ( bvh instanceof SkinnedMeshBVH ) {

				const sourceMesh = bvh.mesh;
				const clonedMesh = new SkinnedMesh( proxyGeometry );
				clonedMesh.copy( sourceMesh );
				clonedMesh.matrixWorld
					.copy( sourceMesh.matrixWorld )
					.decompose( clonedMesh.position, clonedMesh.quaternion, clonedMesh.scale );

				newBVH = new SkinnedMeshBVH( clonedMesh, { strategy: SAH, maxLeafSize: 1 } );

			} else {

				newBVH = new MeshBVH( proxyGeometry, { strategy: SAH, maxLeafSize: 1 } );

			}

			bvhMap.set( bvh, { bvh: newBVH, range: { ...rangeTarget } } );
			return newBVH;

		} else {

			return bvh;

		}

	}

}
