import { wgsl } from 'three/tsl';
import { wgslTagFn } from '../lib/nodes/WGSLTagFnNode.js';
import { constants } from './common.wgsl.js';
import { trimResultStruct, overlapResultStruct } from './structs.wgsl.js';

const { PARALLEL_EPSILON, AREA_EPSILON, DIST_EPSILON, VERTEX_EPSILON } = constants;

// Per-invocation private storage for accumulated overlap intervals.
// Exported so the kernel can include it when reading overlapCount / overlaps.
const MAX_OVERLAPS_PER_EDGE = 512;
export const overlapStorage = wgsl( /* wgsl */`
	const MAX_OVERLAPS : u32 = ${ MAX_OVERLAPS_PER_EDGE }u;
	var<private> overlaps     : array<vec2f, ${ MAX_OVERLAPS_PER_EDGE }>;
	var<private> overlapCount : u32 = 0u;
` );

// Clips the edge (lineStart -> lineEnd) to the portion lying at or below the
// plane of triangle (a, b, c). The plane is always treated as up-facing.
// Returns TrimResult.valid = false if the entire edge is above the plane.
export const trimToBeneathTriPlane = wgslTagFn/* wgsl */`
	fn trimToBeneathTriPlane( a: vec3f, b: vec3f, c: vec3f, lineStart: vec3f, lineEnd: vec3f ) -> ${ trimResultStruct } {

		var result: ${ trimResultStruct };

		// compute the triangle plane, ensuring the normal faces up
		var normal = normalize( cross( b - a, c - a ) );
		if ( normal.y < 0.0 ) {

			normal = -normal;

		}

		let d         = -dot( normal, a );
		let startDist = dot( normal, lineStart ) + d;
		let endDist   = dot( normal, lineEnd ) + d;

		// coplanar/parallel - only valid if the line is below the plane
		let lineDir = normalize( lineEnd - lineStart );
		if ( abs( dot( normal, lineDir ) ) < ${ PARALLEL_EPSILON } ) {

			if ( startDist < 0.0 ) {

				result.start = lineStart;
				result.end   = lineEnd;
				result.valid = true;

			}
			return result;

		}

		let isStartBelow = startDist < 0.0;
		let isEndBelow   = endDist   < 0.0;

		// both below - keep the full edge
		if ( isStartBelow && isEndBelow ) {

			result.start = lineStart;
			result.end   = lineEnd;
			result.valid = true;
			return result;

		}

		// both above - discard
		if ( ! isStartBelow && ! isEndBelow ) {

			return result;

		}

		// straddling - clip at the plane intersection
		let t        = -startDist / ( endDist - startDist );
		let hitPoint = mix( lineStart, lineEnd, t );

		if ( isStartBelow ) {

			result.start = lineStart;
			result.end   = hitPoint;

		} else {

			result.start = hitPoint;
			result.end   = lineEnd;

		}

		result.valid = true;
		return result;

	}
`;

// Returns the parametric overlap [t0, t1] of the edge (lineStart -> lineEnd)
// against triangle (a, b, c) projected onto the XZ plane.
// t0 and t1 are in [0, 1] along the original edge. valid = false if no overlap.
export const getProjectedOverlapRange = wgslTagFn/* wgsl */`
	fn getProjectedOverlapRange( lineStart: vec3f, lineEnd: vec3f, a: vec3f, b: vec3f, c: vec3f ) -> ${ overlapResultStruct } {

		var result: ${ overlapResultStruct };

		// project everything to XZ
		let ls = vec3f( lineStart.x, 0.0, lineStart.z );
		let le = vec3f( lineEnd.x,   0.0, lineEnd.z   );
		let fa = vec3f( a.x, 0.0, a.z );
		let fb = vec3f( b.x, 0.0, b.z );
		let fc = vec3f( c.x, 0.0, c.z );

		// skip degenerate projected triangles
		if ( abs( cross( fb - fa, fc - fa ).y ) <= ${ AREA_EPSILON } ) {

			return result;

		}

		let lineVec = le - ls;
		let lineLen = length( lineVec );
		let dir     = lineVec / lineLen;

		// cutting plane: orthogonal to the edge direction in XZ, passing through ls
		let ortho     = normalize( cross( dir, vec3f( 0.0, 1.0, 0.0 ) ) );
		let planeDist = dot( ortho, ls );

		// find the two intersections of triangle edges with the cutting plane
		var intersectCount = 0u;
		var triLineStart   = vec3f( 0.0 );
		var triLineEnd     = vec3f( 0.0 );

		let triPts = array<vec3f, 3>( fa, fb, fc );
		for ( var i = 0u; i < 3u; i = i + 1u ) {

			let p1 = triPts[ i ];
			let p2 = triPts[ ( i + 1u ) % 3u ];

			let d1 = dot( ortho, p1 ) - planeDist;
			let d2 = dot( ortho, p2 ) - planeDist;

			let startOnPlane = abs( d1 ) < ${ DIST_EPSILON };
			let endOnPlane   = abs( d2 ) < ${ DIST_EPSILON };

			var point        = vec3f( 0.0 );
			var edgeCrossing = false;
			if ( ! startOnPlane && ! endOnPlane && d1 * d2 < 0.0 ) {

				point        = mix( p1, p2, d1 / ( d1 - d2 ) );
				edgeCrossing = true;

			}

			if ( ( edgeCrossing && ! endOnPlane ) || startOnPlane ) {

				if ( startOnPlane && ! edgeCrossing ) {

					point = p1;

				}

				if ( intersectCount == 0u ) {

					triLineStart = point;

				} else {

					triLineEnd = point;

				}

				intersectCount = intersectCount + 1u;
				if ( intersectCount >= 2u ) {

					break;

				}

			}

		}

		if ( intersectCount < 2u ) {

			return result;

		}

		// orient the triangle segment to match the edge direction
		let triSegLen = length( triLineEnd - triLineStart );
		if ( triSegLen < ${ DIST_EPSILON } ) {

			return result;

		}

		var tsStart = triLineStart;
		var tsEnd   = triLineEnd;
		if ( dot( dir, ( triLineEnd - triLineStart ) / triSegLen ) < 0.0 ) {

			tsStart = triLineEnd;
			tsEnd   = triLineStart;

		}

		// project both segments onto dir and compute the overlap
		let s1 = 0.0;
		let e1 = dot( le - ls, dir );
		let s2 = dot( tsStart - ls, dir );
		let e2 = dot( tsEnd   - ls, dir );

		if ( e1 <= s2 || e2 <= s1 ) {

			return result;

		}

		result.t0    = max( s1, s2 ) / lineLen;
		result.t1    = min( e1, e2 ) / lineLen;
		result.valid = true;
		return result;

	}
`;

// Appends a raw [t0, t1] interval to the per-invocation private buffer.
export const appendOverlap = wgslTagFn/* wgsl */`
	${ [ overlapStorage ] }
	fn appendOverlap( t0: f32, t1: f32 ) {

		if ( overlapCount < MAX_OVERLAPS ) {

			overlaps[ overlapCount ] = vec2f( t0, t1 );
			overlapCount = overlapCount + 1u;

		}

	}
`;

// Returns true if the edge (lineStart -> lineEnd) lies entirely along the Y axis
// when projected to XZ — i.e. the line direction is nearly (0, ±1, 0).
export const isYProjectedLineDegenerate = wgslTagFn/* wgsl */`
	fn isYProjectedLineDegenerate( lineStart: vec3f, lineEnd: vec3f ) -> bool {

		let dir = normalize( lineEnd - lineStart );
		return abs( dir.y ) >= 1.0 - ${ VERTEX_EPSILON };

	}
`;

// Returns true if both endpoints of the edge (lineStart -> lineEnd) coincide
// with two vertices of triangle (a, b, c) — i.e. the edge is a triangle edge.
export const isLineTriangleEdge = wgslTagFn/* wgsl */`
	fn isLineTriangleEdge( lineStart: vec3f, lineEnd: vec3f, a: vec3f, b: vec3f, c: vec3f ) -> bool {

		let triPts = array<vec3f, 3>( a, b, c );
		var startMatches = false;
		var endMatches   = false;
		for ( var i = 0u; i < 3u; i = i + 1u ) {

			let tp = triPts[ i ];
			let ds = lineStart - tp;
			let de = lineEnd   - tp;
			if ( ! startMatches && dot( ds, ds ) <= ${ VERTEX_EPSILON } ) {

				startMatches = true;

			}

			if ( ! endMatches && dot( de, de ) <= ${ VERTEX_EPSILON } ) {

				endMatches = true;

			}

			if ( startMatches && endMatches ) {

				return true;

			}

		}

		return startMatches && endMatches;

	}
`;

// Sorts the private overlap buffer by t0 (insertion sort) then merges
// adjacent/overlapping intervals in-place.
export const sortAndMergeOverlaps = wgslTagFn/* wgsl */`
	${ [ overlapStorage ] }
	fn sortAndMergeOverlaps() {

		// insertion sort by t0
		for ( var i = 1u; i < overlapCount; i = i + 1u ) {

			let key = overlaps[ i ];
			var j   = i32( i ) - 1;
			loop {

				if ( j < 0 ) { break; }
				if ( overlaps[ u32( j ) ].x <= key.x ) { break; }
				overlaps[ u32( j + 1 ) ] = overlaps[ u32( j ) ];
				j = j - 1;

			}
			overlaps[ u32( j + 1 ) ] = key;

		}

		// merge overlapping/adjacent intervals
		if ( overlapCount == 0u ) { return; }

		var writeIdx = 0u;
		for ( var i = 1u; i < overlapCount; i = i + 1u ) {

			if ( overlaps[ i ].x <= overlaps[ writeIdx ].y ) {

				overlaps[ writeIdx ].y = max( overlaps[ writeIdx ].y, overlaps[ i ].y );

			} else {

				writeIdx = writeIdx + 1u;
				overlaps[ writeIdx ] = overlaps[ i ];

			}

		}
		overlapCount = writeIdx + 1u;

	}
`;
