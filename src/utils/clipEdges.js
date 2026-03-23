// Clips edges in-place to the region at or below clipY, discarding edges
// entirely above and trimming edges that straddle the boundary.
export function clipEdges( edges, clipY ) {

	let writeIdx = 0;
	for ( let i = 0; i < edges.length; i ++ ) {

		const edge = edges[ i ];
		const aAbove = edge.start.y > clipY;
		const bAbove = edge.end.y > clipY;
		const delta = edge.end.y - edge.start.y;

		if ( aAbove && bAbove ) {

			continue;

		} else if ( aAbove ) {

			const t = ( clipY - edge.start.y ) / delta;
			edge.start.lerp( edge.end, t );

		} else if ( bAbove ) {

			const t = ( clipY - edge.start.y ) / delta;
			edge.end.lerpVectors( edge.start, edge.end, t );

		}

		edges[ writeIdx ++ ] = edge;

	}

	edges.length = writeIdx;

}
