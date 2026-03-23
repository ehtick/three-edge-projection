import { StructTypeNode } from 'three/webgpu';

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
