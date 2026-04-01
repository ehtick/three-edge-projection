/** @import { Object3D } from 'three' */
/** @import { WebGPURenderer } from 'three/webgpu' */
import {
	Box3,
	Vector3,
	Vector4,
	OrthographicCamera,
	Color,
	Mesh,
} from 'three';
import { RenderTarget, MeshBasicNodeMaterial } from 'three/webgpu';
import { uniform } from 'three/tsl';
import { getAllMeshes } from '../utils/getAllMeshes.js';

// RGBA8 ID encoding - supports up to 16,777,215 objects (2^24 - 1)
// ID 0 is valid, background is indicated by alpha = 0
function encodeId( id, target ) {

	target.x = ( id & 0xFF ) / 255;
	target.y = ( ( id >> 8 ) & 0xFF ) / 255;
	target.z = ( ( id >> 16 ) & 0xFF ) / 255;
	target.w = 1;

}

function decodeId( buffer, index ) {

	return buffer[ index ] | ( buffer[ index + 1 ] << 8 ) | ( buffer[ index + 2 ] << 16 );

}

/**
 * Utility for determining visible geometry from a top down orthographic perspective. This can
 * be run before performing projection generation to reduce the complexity of the operation at
 * the cost of potentially missing small details.
 *
 * Takes the WebGPURenderer instance used to render.
 * @param {WebGPURenderer} renderer
 * @param {Object} [options]
 * @param {number} [options.pixelsPerMeter=0.1]
 */
export class MeshVisibilityCuller {

	constructor( renderer, options = {} ) {

		const { pixelsPerMeter = 0.1 } = options;

		/**
		 * The size of a pixel on a single dimension. If this results in a texture larger than what
		 * the graphics context can provide then the rendering is tiled.
		 * @type {number}
		 */
		this.pixelsPerMeter = pixelsPerMeter;
		this.renderer = renderer;

	}

	/**
	 * Returns the set of meshes that are visible within the given object.
	 * @param {Object3D|Array<Object3D>} object
	 * @returns {Promise<Array<Object3D>>}
	 */
	async cull( objects ) {

		objects = getAllMeshes( objects );

		const { renderer, pixelsPerMeter } = this;
		const size = new Vector3();
		const camera = new OrthographicCamera();
		const box = new Box3();

		const idValue = new Vector4();
		const idUniform = uniform( idValue );
		const idMaterial = new MeshBasicNodeMaterial();
		idMaterial.colorNode = idUniform;

		const idMesh = new Mesh( undefined, idMaterial );
		idMesh.matrixAutoUpdate = false;
		idMesh.matrixWorldAutoUpdate = false;

		// get the bounds of the image
		box.makeEmpty();
		objects.forEach( o => {

			box.expandByObject( o );

		} );

		// get the bounds dimensions
		box.getSize( size );

		// calculate the tile and target size
		const maxTextureSize = Math.min( renderer.backend.device.limits.maxTextureDimension2D, 2 ** 13 );
		const pixelWidth = Math.ceil( size.x / pixelsPerMeter );
		const pixelHeight = Math.ceil( size.z / pixelsPerMeter );
		const tilesX = Math.ceil( pixelWidth / maxTextureSize );
		const tilesY = Math.ceil( pixelHeight / maxTextureSize );

		const target = new RenderTarget( Math.ceil( pixelWidth / tilesX ), Math.ceil( pixelHeight / tilesY ) );

		// set the camera bounds
		camera.rotation.x = - Math.PI / 2;
		camera.far = ( box.max.y - box.min.y ) + camera.near;
		camera.position.y = box.max.y + camera.near;

		// save render state
		const color = renderer.getClearColor( new Color() );
		const alpha = renderer.getClearAlpha();
		const renderTarget = renderer.getRenderTarget();
		const autoClear = renderer.autoClear;

		// render ids
		const visibleSet = new Set();
		const stepX = size.x / tilesX;
		const stepZ = size.z / tilesY;
		for ( let x = 0; x < tilesX; x ++ ) {

			for ( let y = 0; y < tilesY; y ++ ) {

				// update camera
				camera.left = box.min.x + stepX * x;
				camera.top = - ( box.min.z + stepZ * y );

				camera.right = camera.left + stepX;
				camera.bottom = camera.top - stepZ;

				camera.updateProjectionMatrix();

				// clear the render target
				renderer.autoClear = false;
				renderer.setClearColor( 0, 0 );
				renderer.setRenderTarget( target );
				renderer.clear();

				for ( let i = 0; i < objects.length; i ++ ) {

					const object = objects[ i ];
					idMesh.matrixWorld.copy( object.matrixWorld );
					idMesh.geometry = object.geometry;

					encodeId( i, idValue );
					renderer.render( idMesh, camera );

				}

				// reset render state before async operation to avoid corruption
				renderer.setClearColor( color, alpha );
				renderer.setRenderTarget( renderTarget );
				renderer.autoClear = autoClear;

				const buffer = new Uint8Array( await renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height ) );

				// find all visible objects - decode RGBA to ID
				for ( let i = 0, l = buffer.length; i < l; i += 4 ) {

					// alpha = 0 indicates background (no object)
					if ( buffer[ i + 3 ] === 0 ) continue;

					const id = decodeId( buffer, i );
					visibleSet.add( objects[ id ] );

				}

			}

		}

		// dispose of intermediate values
		idMaterial.dispose();
		target.dispose();

		return Array.from( visibleSet );

	}

}
