import { StructTypeNode } from 'three/webgpu';

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
};

export const PlaneWGSL = {
	struct: planeStruct,
};
