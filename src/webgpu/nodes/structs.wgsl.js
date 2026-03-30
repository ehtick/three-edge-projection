import { StructTypeNode } from 'three/webgpu';

export const edgeStruct = new StructTypeNode( {
	start: 'array<f32, 3>',
	end: 'array<f32, 3>',
	index: 'uint',
}, 'Edge' );
edgeStruct.getLength = () => 7;

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
	_alignment0: 'uint',
}, 'TriEdgePair' );

// One entry per visible overlap interval recorded during kernel 3.
export const overlapRecordStruct = new StructTypeNode( {
	edgeIndex: 'uint',
	t0: 'float',
	t1: 'float',
	_alignment0: 'uint',
}, 'OverlapRecord' );
