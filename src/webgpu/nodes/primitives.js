import { StructTypeNode } from 'three/webgpu';
import { wgslTagFn } from '../lib/nodes/WGSLTagFnNode.js';

const lineStruct = new StructTypeNode( {
	start: 'vec3',
	end: 'vec3',
} );

const triStruct = new StructTypeNode( {
	a: 'vec3',
	b: 'vec3',
	c: 'vec3',
} );

const planeStruct = new StructTypeNode( {
	normal: 'vec3',
	constant: 'float',
} );

export const LineWGSL = {
	struct: lineStruct,
};

export const TriWGSL = {
	struct: triStruct,
	getNormal: wgslTagFn/* wgsl */`
		fn tri_getNormal( tri: ${ triStruct } ) -> vec3f {

			return normalize( cross( tri.c - tri.b, tri.a - tri.b ) );

		}
	`,
	getArea: wgslTagFn/* wgsl */`
		fn tri_getArea( tri: ${ triStruct } ) -> f32 {

			return length( cross( tri.c - tri.b, tri.a - tri.b ) * 0.5 );

		}
	`
};

export const PlaneWGSL = {
	struct: planeStruct,
	fromNormalAndCoplanarPoint: wgslTagFn/* wgsl */`
		fn plane_fromNormalAndCoplanarPoint( norm: vec3f, point: vec3f ) -> ${ planeStruct } {

			var plane: ${ planeStruct };
			plane.normal = norm;
			plane.constant = - dot( norm, point );
			return plane;

		}
	`,
};
