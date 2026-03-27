import { StructTypeNode } from 'three/webgpu';

export const internalEdge = new StructTypeNode( {
	start: 'vec3',
	end: 'vec3',
} );

export const internalTri = new StructTypeNode( {
	a: 'vec3',
	b: 'vec3',
	c: 'vec3',
} );

export const edgeStruct = new StructTypeNode( {
	start: 'array<f32, 3>',
	end: 'array<f32, 3>',
	index: 'uint',
}, 'Edge' );
edgeStruct.getLength = () => 7;

export const trimResultStruct = new StructTypeNode( {
	start: 'vec3f',
	end: 'vec3f',
	valid: 'bool',
}, 'TrimResult' );

export const overlapResultStruct = new StructTypeNode( {
	t0: 'f32',
	t1: 'f32',
	valid: 'bool',
}, 'OverlapResult' );

export const clipResultStruct = new StructTypeNode( {
	count: 'uint',
	a0: 'vec3f',
	b0: 'vec3f',
	c0: 'vec3f',
	a1: 'vec3f',
	b1: 'vec3f',
	c1: 'vec3f',
}, 'ClipResult' );

// One entry per qualifying (edge, triangle) pair recorded during kernel 2.
export const triEdgePairStruct = new StructTypeNode( {
	edgeIndex: 'uint',
	objectIndex: 'uint',
	triIndex: 'uint',
}, 'TriEdgePair' );
triEdgePairStruct.getLength = () => 3;

// One entry per visible overlap interval recorded during kernel 3.
export const overlapRecordStruct = new StructTypeNode( {
	edgeIndex: 'uint',
	t0: 'f32',
	t1: 'f32',
}, 'OverlapRecord' );
overlapRecordStruct.getLength = () => 3;
