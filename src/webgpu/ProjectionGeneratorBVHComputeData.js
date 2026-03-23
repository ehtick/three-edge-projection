import { BufferAttribute, BufferGeometry, SkinnedMesh } from 'three';
import { StructTypeNode } from 'three/webgpu';
import { BVHComputeData } from './lib/BVHComputeData.js';
import { MeshBVH, SAH, SkinnedMeshBVH } from 'three-mesh-bvh';
import { wgslTagFn } from './lib/nodes/WGSLTagFnNode.js';
import { bvhNodeStruct, bvhNodeBoundsStruct } from './lib/wgsl/structs.wgsl.js';
import { constants as overlapConstants } from './nodes/common.wgsl.js';
import {
	appendOverlap,
	sortAndMergeOverlaps,
	trimToBeneathTriPlane,
	getProjectedOverlapRange,
	isLineTriangleEdge,
} from './nodes/overlapFunctions.wgsl.js';

// Shape struct carrying both world-space and local-space line endpoints,
// plus the transform buffer index (set by transformShapeFn during TLAS traversal).
const edgeLineShapeStruct = new StructTypeNode( {
	worldStart: 'vec3f',
	worldEnd: 'vec3f',
	localStart: 'vec3f',
	localEnd: 'vec3f',
	objectIndex: 'uint',
}, 'EdgeLineShape' );

// Minimal result struct satisfying the shapecast interface.
// We never set didHit = true, so no best-hit pruning occurs.
const edgeOverlapResultStruct = new StructTypeNode( {
	didHit: 'bool',
	dist: 'float',
}, 'EdgeOverlapResult' );

// Projection-generator-specific BVHComputeData that only requires position
// attributes and auto-generates missing BVHs.
export class ProjectionGeneratorBVHComputeData extends BVHComputeData {

	constructor( bvh, options = {} ) {

		super( bvh, {
			attributes: { position: 'vec4f' },
			...options,
		} );

		this.bvhMap = new Map();

	}

	update() {

		super.update();
		this.bvhMap.clear();

	}

	getOverlapsFn( name ) {

		const { storage } = this;
		const { DIST_EPSILON } = overlapConstants;

		const boundsOrderFn = wgslTagFn/* wgsl */`
			fn ${ name }BoundsOrder( shape: ${ edgeLineShapeStruct }, splitAxis: u32, node: ${ bvhNodeStruct } ) -> bool {

				return true;

			}
		`;

		const intersectsBoundsFn = wgslTagFn/* wgsl */`
			fn ${ name }IntersectsBounds( shape: ${ edgeLineShapeStruct }, bounds: ${ bvhNodeBoundsStruct } ) -> f32 {

				// Y-cull using world-space line endpoints; bounds.max[1] is in local
				// space at BLAS level but the approximation is acceptable.
				if ( bounds.max[ 1 ] <= min( shape.worldStart.y, shape.worldEnd.y ) ) {

					return -1.0;

				}

				// XZ AABB cull — localStart/End equals worldStart/End at TLAS level,
				// and is local-space after transformShapeFn is applied for BLAS.
				let localMinX = min( shape.localStart.x, shape.localEnd.x );
				let localMaxX = max( shape.localStart.x, shape.localEnd.x );
				let localMinZ = min( shape.localStart.z, shape.localEnd.z );
				let localMaxZ = max( shape.localStart.z, shape.localEnd.z );
				if ( bounds.max[ 0 ] < localMinX || bounds.min[ 0 ] > localMaxX ||
				     bounds.max[ 2 ] < localMinZ || bounds.min[ 2 ] > localMaxZ ) {

					return -1.0;

				}

				return 0.0;

			}
		`;

		const intersectRangeFn = wgslTagFn/* wgsl */`
			fn ${ name }IntersectRange( shape: ${ edgeLineShapeStruct }, offset: u32, count: u32, bestDist: f32 ) -> ${ edgeOverlapResultStruct } {

				var result: ${ edgeOverlapResultStruct };
				result.didHit = false;
				result.dist   = bestDist;

				let lineWorldStart = shape.worldStart;
				let lineWorldEnd   = shape.worldEnd;
				let lineMinY = min( lineWorldStart.y, lineWorldEnd.y );
				let lineDir  = normalize( lineWorldEnd - lineWorldStart );
				let lineLen  = length( lineWorldEnd - lineWorldStart );

				for ( var ti = offset; ti < offset + count; ti = ti + 1u ) {

					let i0 = ${ storage.index }[ ti * 3u ];
					let i1 = ${ storage.index }[ ti * 3u + 1u ];
					let i2 = ${ storage.index }[ ti * 3u + 2u ];

					let localA = ${ storage.attributes }[ i0 ].position.xyz;
					let localB = ${ storage.attributes }[ i1 ].position.xyz;
					let localC = ${ storage.attributes }[ i2 ].position.xyz;

					// transform triangle vertices to world space
					let matrixWorld = ${ storage.transforms }[ shape.objectIndex ].matrixWorld;
					let a = ( matrixWorld * vec4f( localA, 1.0 ) ).xyz;
					let b = ( matrixWorld * vec4f( localB, 1.0 ) ).xyz;
					let c = ( matrixWorld * vec4f( localC, 1.0 ) ).xyz;

					// skip triangles entirely below the edge
					let highestTriY = max( max( a.y, b.y ), c.y );
					if ( highestTriY <= lineMinY ) {

						continue;

					}

					// skip if the edge lies on this triangle
					if ( ${ isLineTriangleEdge }( lineWorldStart, lineWorldEnd, a, b, c ) ) {

						continue;

					}

					// trim edge to the portion below the triangle plane
					let trimResult = ${ trimToBeneathTriPlane }( a, b, c, lineWorldStart, lineWorldEnd );
					if ( ! trimResult.valid ) {

						continue;

					}

					// skip degenerate trimmed segments
					let trimLen = length( trimResult.end - trimResult.start );
					if ( trimLen < ${ DIST_EPSILON } ) {

						continue;

					}

					// get projected overlap range in trimmed-edge space
					let overlapRange = ${ getProjectedOverlapRange }( trimResult.start, trimResult.end, a, b, c );
					if ( ! overlapRange.valid ) {

						continue;

					}

					// remap t values from trimmed-edge space to original-edge space
					let tTrimStart = dot( trimResult.start - lineWorldStart, lineDir ) / lineLen;
					let tTrimEnd   = dot( trimResult.end   - lineWorldStart, lineDir ) / lineLen;
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
			fn ${ name }TransformShape( shape: ${ edgeLineShapeStruct }, inverseMatrix: mat4x4f, objectIndex: u32 ) -> ${ edgeLineShapeStruct } {

				var localShape = shape;
				localShape.localStart  = ( inverseMatrix * vec4f( shape.worldStart, 1.0 ) ).xyz;
				localShape.localEnd    = ( inverseMatrix * vec4f( shape.worldEnd,   1.0 ) ).xyz;
				localShape.objectIndex = objectIndex;
				return localShape;

			}
		`;

		const transformResultFn = wgslTagFn/* wgsl */`
			fn ${ name }TransformResult( result: ptr<function, ${ edgeOverlapResultStruct }>, objectIndex: u32 ) -> void {

			}
		`;

		const traversalFn = this.getShapecastFn( {
			name: name + '_traversal',
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
		const overlapsFn = wgslTagFn/* wgsl */`
			fn ${ name }( lineStart: vec3f, lineEnd: vec3f ) {

				var shape: ${ edgeLineShapeStruct };
				shape.worldStart = lineStart;
				shape.worldEnd   = lineEnd;
				shape.localStart = lineStart;
				shape.localEnd   = lineEnd;
				${ traversalFn }( shape );
				${ sortAndMergeOverlaps }();

			}
		`;

		return overlapsFn;

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
