import { StructTypeNode } from 'three/webgpu';

export const edgeStruct = new StructTypeNode( {
	start: 'array<f32, 3>',
	end: 'array<f32, 3>',
	index: 'uint',
}, 'Edge' );
edgeStruct.getLength = () => 7;
