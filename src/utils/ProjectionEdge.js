import { Line3 } from 'three';

export class ProjectionEdge extends Line3 {

	constructor( start, end ) {

		super( start, end );
		this.mesh = null;

	}

	copy( source ) {

		super.copy( source );
		this.mesh = source.mesh || null;
		return this;

	}

}
