import { float, int } from 'three/tsl';

export const constants = {
	PARALLEL_EPSILON: float( 1e-10 ),
	AREA_EPSILON: float( 1e-10 ),
	DIST_EPSILON: float( 1e-10 ),
	VERTEX_EPSILON: float( 1e-10 ),

	DOUBLE_SIDE: int( 0 ),
	BACK_SIDE: int( - 1 ),
	FRONT_SIDE: int( 1 ),
};
