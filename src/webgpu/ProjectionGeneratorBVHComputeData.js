import { BackSide, BufferAttribute, BufferGeometry, DoubleSide, FrontSide, SkinnedMesh } from 'three';
import { FrontFacingNode, StructTypeNode } from 'three/webgpu';
import { BVHComputeData } from './lib/BVHComputeData.js';
import { MeshBVH, SAH, SkinnedMeshBVH } from 'three-mesh-bvh';
import { wgslTagFn } from './lib/nodes/WGSLTagFnNode.js';
import { bvhNodeStruct, bvhNodeBoundsStruct } from './lib/wgsl/structs.wgsl.js';
import { transformBVHBounds } from './nodes/utils.wgsl.js';
import { constants as overlapConstants } from './nodes/common.wgsl.js';
import {
	appendOverlap,
	sortAndMergeOverlaps,
	trimToBeneathTriPlane,
	getProjectedOverlapRange,
	isLineTriangleEdge,
} from './nodes/overlapFunctions.wgsl.js';

// Shape struct carrying world-space line endpoints plus the object-to-world
// matrix (set by transformShapeFn; identity at top level so world-space
// bounds pass through unchanged) and the transform buffer index.
const edgeLineShapeStruct = new StructTypeNode( {
	worldStart: 'vec3f',
	worldEnd: 'vec3f',
	matrixWorld: 'mat4x4f',
	objectIndex: 'uint',
}, 'EdgeLineShape' );

// Minimal result struct satisfying the shapecast interface.
// We never set didHit = true, so no best-hit pruning occurs.
const edgeOverlapResultStruct = new StructTypeNode( {
	didHit: 'bool',
	dist: 'float',
}, 'EdgeOverlapResult' );

// Extended transform struct that adds a per-object "side" field for back-face
// culling. Layout matches the base struct; "side" reuses the _alignment0 slot.
// Values: 0 = DoubleSide (no cull), 1 = FrontSide, 2 = BackSide.
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

	}

	getOverlapsFn() {

		const { storage, prefix } = this;
		const { DIST_EPSILON } = overlapConstants;

		const boundsOrderFn = wgslTagFn/* wgsl */`
			fn ${ prefix }BoundsOrder( shape: ${ edgeLineShapeStruct }, splitAxis: u32, node: ${ bvhNodeStruct } ) -> bool {

				return true;

			}
		`;

		const intersectsBoundsFn = wgslTagFn/* wgsl */`
			fn ${ prefix }IntersectsBounds( shape: ${ edgeLineShapeStruct }, bounds: ${ bvhNodeBoundsStruct } ) -> f32 {

				// Transform bounds to world space. At the top level the shape matrix
				// is identity, so world-space bounds pass through unchanged.
				let aabb = ${ transformBVHBounds }( bounds, shape.matrixWorld );

				// Y-cull: bounds entirely below the line
				if ( aabb.max[ 1 ] <= min( shape.worldStart.y, shape.worldEnd.y ) ) {

					return - 1.0;

				}

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

		const intersectRangeFn = wgslTagFn/* wgsl */`
			fn ${ prefix }IntersectRange( shape: ${ edgeLineShapeStruct }, offset: u32, count: u32, bestDist: f32 ) -> ${ edgeOverlapResultStruct } {

				var result: ${ edgeOverlapResultStruct };
				result.didHit = false;
				result.dist = bestDist;

				let lineWorldStart = shape.worldStart;
				let lineWorldEnd = shape.worldEnd;
				let lineMinY = min( lineWorldStart.y, lineWorldEnd.y );
				let lineMaxY = max( lineWorldStart.y, lineWorldEnd.y );
				let lineDir = normalize( lineWorldEnd - lineWorldStart );
				let lineLen = length( lineWorldEnd - lineWorldStart );
				let matrixWorld = shape.matrixWorld;
				let inverted = determinant( matrixWorld ) < 0.0;

				for ( var ti = offset; ti < offset + count; ti = ti + 1u ) {

					let i0 = ${ storage.index }[ ti * 3u ];
					let i1 = ${ storage.index }[ ti * 3u + 1u ];
					let i2 = ${ storage.index }[ ti * 3u + 2u ];

					let localA = ${ storage.attributes }[ i0 ].position.xyz;
					let localB = ${ storage.attributes }[ i1 ].position.xyz;
					let localC = ${ storage.attributes }[ i2 ].position.xyz;

					let a = ( matrixWorld * vec4f( localA, 1.0 ) ).xyz;
					let b = ( matrixWorld * vec4f( localB, 1.0 ) ).xyz;
					let c = ( matrixWorld * vec4f( localC, 1.0 ) ).xyz;

					// back-face cull based on per-object side (0=double, 1=front, -1=back)
					let triNormal = cross( b - a, c - a );
					let side = ${ storage.transforms }[ shape.objectIndex ].side;
					if ( side != 0 ) {

						let isFrontUp = ( triNormal.y > 0.0 ) != inverted;
						let keepFront = side > 0;
						if ( isFrontUp != keepFront ) {

							continue;

						}

					}

					// skip triangles entirely below the edge
					let highestTriY = max( max( a.y, b.y ), c.y );
					if ( highestTriY <= lineMinY ) {

						continue;

					}

					// skip if the edge lies on this triangle
					if ( ${ isLineTriangleEdge }( lineWorldStart, lineWorldEnd, a, b, c ) ) {

						continue;

					}

					// trim edge to the portion below the triangle plane; if the
					// entire line is already below the triangle, use the full line
					let lowestTriY = min( min( a.y, b.y ), c.y );
					var trimStart = lineWorldStart;
					var trimEnd   = lineWorldEnd;
					if ( lineMaxY >= lowestTriY ) {

						let trimResult = ${ trimToBeneathTriPlane }( a, b, c, lineWorldStart, lineWorldEnd );
						if ( ! trimResult.valid ) {

							continue;

						}

						trimStart = trimResult.start;
						trimEnd   = trimResult.end;

					}

					// skip degenerate trimmed segments
					let trimLen = length( trimEnd - trimStart );
					if ( trimLen < ${ DIST_EPSILON } ) {

						continue;

					}

					// get projected overlap range in trimmed-edge space
					let overlapRange = ${ getProjectedOverlapRange }( trimStart, trimEnd, a, b, c );
					if ( ! overlapRange.valid ) {

						continue;

					}

					// remap t values from trimmed-edge space to original-edge space
					let tTrimStart = dot( trimStart - lineWorldStart, lineDir ) / lineLen;
					let tTrimEnd   = dot( trimEnd   - lineWorldStart, lineDir ) / lineLen;
					let t0 = clamp( tTrimStart + overlapRange.t0 * ( tTrimEnd - tTrimStart ), 0.0, 1.0 );
					let t1 = clamp( tTrimStart + overlapRange.t1 * ( tTrimEnd - tTrimStart ), 0.0, 1.0 );
					if ( t0 < t1 ) {

						${ appendOverlap }( t0, t1 );

					}

				}

				return result;

			}
		`;

		const transformShapeFn = wgslTagFn/* wgsl */`
			fn ${ prefix }TransformShape( shape: ${ edgeLineShapeStruct }, inverseMatrix: mat4x4f, objectIndex: u32 ) -> ${ edgeLineShapeStruct } {

				var localShape = shape;
				localShape.matrixWorld = ${ storage.transforms }[ objectIndex ].matrixWorld;
				localShape.objectIndex = objectIndex;
				return localShape;

			}
		`;

		const transformResultFn = wgslTagFn/* wgsl */`
			fn ${ prefix }TransformResult( result: ptr<function, ${ edgeOverlapResultStruct }>, objectIndex: u32 ) -> void {

			}
		`;

		const traversalFn = this.getShapecastFn( {
			name: prefix + '_traversal',
			shapeStruct: edgeLineShapeStruct,
			resultStruct: edgeOverlapResultStruct,
			boundsOrderFn,
			intersectsBoundsFn,
			intersectRangeFn,
			transformShapeFn,
			transformResultFn,
		} );

		// Thin wrapper: packs the line into an EdgeLineShape, runs the full
		// TLAS/BLAS traversal (accumulating into the private buffer), then
		// sorts and merges the resulting intervals.
		return wgslTagFn/* wgsl */`
			fn ${ prefix }( lineStart: vec3f, lineEnd: vec3f ) {

				var shape: ${ edgeLineShapeStruct };
				shape.worldStart  = lineStart;
				shape.worldEnd    = lineEnd;
				shape.matrixWorld = mat4x4f( 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 );
				shape.objectIndex = 0u;
				${ traversalFn }( shape );
				${ sortAndMergeOverlaps }();

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
