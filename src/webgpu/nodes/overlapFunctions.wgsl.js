import { wgslTagFn } from '../lib/nodes/WGSLTagFnNode.js';
import { constants } from './common.wgsl.js';
import { overlapResultStruct, clipResultStruct, internalTri, internalEdge } from './structs.wgsl.js';

const { PARALLEL_EPSILON, AREA_EPSILON, DIST_THRESHOLD: DIST_EPSILON, VERTEX_EPSILON } = constants;

// Clips triangle (a, b, c) against a plane (plane.xyz = normal, plane.w = constant,
// equation: dot(normal, p) + constant >= 0 is the kept side).
// Returns 0, 1, or 2 sub-triangles covering the kept portion.
export const clipTriangleToPlane = wgslTagFn/* wgsl */`
	fn clipTriangleToPlane( a: vec3f, b: vec3f, c: vec3f, plane: vec4f ) -> ${ clipResultStruct } {

		var result: ${ clipResultStruct };

		let da = dot( plane.xyz, a ) + plane.w;
		let db = dot( plane.xyz, b ) + plane.w;
		let dc = dot( plane.xyz, c ) + plane.w;

		let aKept = da >= 0.0;
		let bKept = db >= 0.0;
		let cKept = dc >= 0.0;
		let keptCount = u32( aKept ) + u32( bKept ) + u32( cKept );

		// all kept - return the original triangle
		if ( keptCount == 3u ) {

			result.count = 1u;
			result.a0 = a;
			result.b0 = b;
			result.c0 = c;
			return result;

		}

		// all discarded
		if ( keptCount == 0u ) {

			return result;

		}

		// vertex positions and plane distances packed into arrays for index-based access
		let pts   = array<vec3f, 3>( a, b, c );
		let dists = array<f32,   3>( da, db, dc );

		if ( keptCount == 1u ) {

			// apex is the lone kept vertex; the other two are clipped away
			var apexIdx = 0u;
			if ( bKept ) {

				apexIdx = 1u;

			} else if ( cKept ) {

				apexIdx = 2u;

			}

			let apex     = pts[ apexIdx ];
			let clipped0 = pts[ ( apexIdx + 1u ) % 3u ];
			let clipped1 = pts[ ( apexIdx + 2u ) % 3u ];

			let apexDist     = dists[ apexIdx ];
			let clipped0Dist = dists[ ( apexIdx + 1u ) % 3u ];
			let clipped1Dist = dists[ ( apexIdx + 2u ) % 3u ];

			// parametric intersection along apex->clipped0 and apex->clipped1
			let t0 = apexDist / ( apexDist - clipped0Dist );
			let t1 = apexDist / ( apexDist - clipped1Dist );

			result.count = 1u;
			result.a0 = apex;
			result.b0 = mix( apex, clipped0, t0 );
			result.c0 = mix( apex, clipped1, t1 );
			return result;

		}

		// the lone discarded vertex is cut off, leaving a quad that we split into two triangles
		var discardedIdx = 2u;
		if ( ! aKept ) {

			discardedIdx = 0u;

		} else if ( ! bKept ) {

			discardedIdx = 1u;

		}

		// kept0 and kept1 are the two vertices on the kept side; discarded is the one being cut off
		let kept0     = pts[ ( discardedIdx + 1u ) % 3u ];
		let kept1     = pts[ ( discardedIdx + 2u ) % 3u ];
		let discarded = pts[ discardedIdx ];

		let kept0Dist     = dists[ ( discardedIdx + 1u ) % 3u ];
		let kept1Dist     = dists[ ( discardedIdx + 2u ) % 3u ];
		let discardedDist = dists[ discardedIdx ];

		// parametric intersections along kept0->discarded and kept1->discarded
		let t0 = kept0Dist / ( kept0Dist - discardedDist );
		let t1 = kept1Dist / ( kept1Dist - discardedDist );

		let edge0Cut = mix( kept0, discarded, t0 );
		let edge1Cut = mix( kept1, discarded, t1 );

		// quad (kept0, kept1, edge1Cut, edge0Cut) split into two triangles
		result.count = 2u;
		result.a0 = kept0;
		result.b0 = kept1;
		result.c0 = edge1Cut;
		result.a1 = kept0;
		result.b1 = edge1Cut;
		result.c1 = edge0Cut;
		return result;

	}
`;

// Clips the edge (lineStart -> lineEnd) to the portion lying at or below the
// plane of triangle (a, b, c). The plane is always treated as up-facing.
// Returns TrimResult.valid = false if the entire edge is above the plane.
export const trimToBeneathTriPlane = wgslTagFn/* wgsl */`
	fn trimToBeneathTriPlane( tri: ${ internalTri }, line: ${ internalEdge }, output: ptr<function, ${ internalEdge }> ) -> bool {

		// TODO: this function seems to be causing issues
		let a = tri.a;
		let b = tri.b;
		let c = tri.c;

		let lineStart = line.start;
		let lineEnd = line.end;

		// compute the triangle plane, ensuring the normal faces up
		var normal = normalize( cross( b - a, c - a ) );
		if ( normal.y < 0.0 ) {

			normal = - normal;

		}

		let d = - dot( normal, a );
		let startDist = dot( normal, lineStart ) + d;
		let endDist = dot( normal, lineEnd ) + d;

		// coplanar/parallel - only valid if the line is below the plane
		let lineDir = normalize( lineEnd - lineStart );
		if ( abs( dot( normal, lineDir ) ) < ${ PARALLEL_EPSILON } ) {

			if ( startDist < 0.0 ) {

				output.start = lineStart;
				output.end = lineEnd;
				return true;

			} else {

				return false;

			}

		}

		let isStartBelow = startDist < 0.0;
		let isEndBelow = endDist   < 0.0;

		// both below - keep the full edge
		if ( isStartBelow && isEndBelow ) {

			output.start = lineStart;
			output.end = lineEnd;
			return true;

		}

		// both above - discard
		if ( ! isStartBelow && ! isEndBelow ) {

			return false;

		}

		// straddling - clip at the plane intersection
		let t = - startDist / ( endDist - startDist );
		let hitPoint = mix( lineStart, lineEnd, t );

		if ( isStartBelow ) {

			output.start = lineStart;
			output.end = hitPoint;

		} else {

			output.start = hitPoint;
			output.end = lineEnd;

		}

		return true;

	}
`;

// Returns the parametric overlap [t0, t1] of the edge (lineStart -> lineEnd)
// against triangle (a, b, c) projected onto the XZ plane.
// t0 and t1 are in [0, 1] along the original edge. valid = false if no overlap.
export const getProjectedOverlapRange = wgslTagFn/* wgsl */`
	fn getProjectedOverlapRange( line: ${ internalEdge }, tri: ${ internalTri } ) -> ${ overlapResultStruct } {

		let a = tri.a;
		let b = tri.b;
		let c = tri.c;

		let lineStart = line.start;
		let lineEnd = line.end;

		var result: ${ overlapResultStruct };

		// project everything to XZ
		let ls = vec3f( lineStart.x, 0.0, lineStart.z );
		let le = vec3f( lineEnd.x, 0.0, lineEnd.z );
		let fa = vec3f( a.x, 0.0, a.z );
		let fb = vec3f( b.x, 0.0, b.z );
		let fc = vec3f( c.x, 0.0, c.z );

		// skip degenerate projected triangles
		// TODO: Add degenerate triangle test function.
		if ( abs( cross( fb - fa, fc - fa ).y ) <= ${ AREA_EPSILON } ) {

			return result;

		}

		let lineVec = le - ls;
		let lineLen = length( lineVec );
		let dir = lineVec / lineLen;

		// cutting plane: orthogonal to the edge direction in XZ, passing through ls
		let ortho = normalize( cross( dir, vec3f( 0.0, 1.0, 0.0 ) ) );
		let planeDist = dot( ortho, ls );

		// find the two intersections of triangle edges with the cutting plane
		var intersectCount = 0u;
		var triLineStart = vec3f( 0.0 );
		var triLineEnd = vec3f( 0.0 );

		let triPts = array<vec3f, 3>( fa, fb, fc );
		for ( var i = 0u; i < 3u; i = i + 1u ) {

			let p1 = triPts[ i ];
			let p2 = triPts[ ( i + 1u ) % 3u ];

			let d1 = dot( ortho, p1 ) - planeDist;
			let d2 = dot( ortho, p2 ) - planeDist;

			let startOnPlane = abs( d1 ) < ${ DIST_EPSILON };
			let endOnPlane = abs( d2 ) < ${ DIST_EPSILON };

			var point = vec3f( 0.0 );
			var edgeCrossing = false;
			if ( ! startOnPlane && ! endOnPlane && d1 * d2 < 0.0 ) {

				point = mix( p1, p2, d1 / ( d1 - d2 ) );
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
		var tsEnd = triLineEnd;
		if ( dot( dir, ( triLineEnd - triLineStart ) / triSegLen ) < 0.0 ) {

			tsStart = triLineEnd;
			tsEnd = triLineStart;

		}

		// project both segments onto dir and compute the overlap
		let s1 = 0.0;
		let e1 = dot( le - ls, dir );
		let s2 = dot( tsStart - ls, dir );
		let e2 = dot( tsEnd - ls, dir );

		if ( e1 <= s2 || e2 <= s1 ) {

			return result;

		}

		result.t0 = max( s1, s2 ) / lineLen;
		result.t1 = min( e1, e2 ) / lineLen;
		result.valid = true;
		return result;

	}
`;


// Returns true if the edge (lineStart -> lineEnd) lies entirely along the Y axis
// when projected to XZ — i.e. the line direction is nearly (0, ±1, 0).
export const isYProjectedLineDegenerate = wgslTagFn/* wgsl */`
	fn isYProjectedLineDegenerate( lineStart: vec3f, lineEnd: vec3f ) -> bool {

		// TODO: just measure the projected distance here
		let dir = normalize( lineEnd - lineStart );
		return abs( dir.y ) >= 1.0 - ${ VERTEX_EPSILON };

	}
`;

// Returns true if both endpoints of the edge (lineStart -> lineEnd) coincide
// with two vertices of triangle (a, b, c) — i.e. the edge is a triangle edge.
export const isLineTriangleEdge = wgslTagFn/* wgsl */`
	fn isLineTriangleEdge( tri: ${ internalTri }, line: ${ internalEdge } ) -> bool {

		let triPts = array<vec3f, 3>( tri.a, tri.b, tri.c );
		var startMatches = false;
		var endMatches   = false;
		for ( var i = 0u; i < 3u; i = i + 1u ) {

			let tp = triPts[ i ];
			let ds = line.start - tp;
			let de = line.end - tp;
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
