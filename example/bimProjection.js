import {
	Quaternion,
	AxesHelper,
	Group,
	MeshLambertMaterial,
	BufferGeometry,
	Float32BufferAttribute,
	Mesh,
	Box3,
	Vector3,
	PlaneGeometry,
	MeshBasicMaterial,
	LineBasicMaterial,
	LineSegments,
	LineDashedMaterial,
} from 'three';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import * as OBC from '@thatopen/components';
import { WebGPURenderer } from 'three/webgpu';
import { ProjectionGenerator, MeshVisibilityCuller } from 'three-edge-projection/webgpu';

const params = {
	displayModel: true,
	displayDrawThroughProjection: false,
	includeIntersectionEdges: false,
	rotate: () => {

		const randomQuaternion = new Quaternion();
		randomQuaternion.random();

		allMeshes.quaternion.copy( randomQuaternion );
		allMeshes.position.set( 0, 0, 0 );
		allMeshes.updateMatrixWorld( true );

		model.object.quaternion.copy( randomQuaternion );
		model.object.position.set( 0, 0, 0 );
		model.object.updateMatrixWorld( true );
		fragments.core.update( true );

	},
	regenerate: () => {

		updateEdges();

	},
};

const ANGLE_THRESHOLD = 50;
let gui;
let projection, drawThroughProjection;
let outputContainer;

// Separate WebGPU renderer for compute (OBC's renderer is WebGL)
const gpuRenderer = new WebGPURenderer();
await gpuRenderer.init();

const components = new OBC.Components();
const worlds = components.get( OBC.Worlds );
const container = document.getElementById( 'container' );

const world = worlds.create();

world.scene = new OBC.SimpleScene( components );
world.renderer = new OBC.SimpleRenderer( components, container );
world.camera = new OBC.OrthoPerspectiveCamera( components );

components.init();

world.scene.setup();

world.scene.three.add( new AxesHelper() );

outputContainer = document.getElementById( 'output' );

const githubUrl = 'https://thatopen.github.io/engine_fragment/resources/worker.mjs';
const fetchedUrl = await fetch( githubUrl );
const workerBlob = await fetchedUrl.blob();
const workerFile = new File( [ workerBlob ], 'worker.mjs', { type: 'text/javascript' } );
const workerUrl = URL.createObjectURL( workerFile );
const fragments = components.get( OBC.FragmentsManager );
fragments.init( workerUrl );

world.camera.controls.addEventListener( 'control', () =>
	fragments.core.update( true ),
);

// Remove z fighting
fragments.core.models.materials.list.onItemSet.add( ( { value: material } ) => {

	if ( ! ( 'isLodMaterial' in material && material.isLodMaterial ) ) {

		material.polygonOffset = true;
		material.polygonOffsetUnits = 1;
		material.polygonOffsetFactor = Math.random();

	}

} );

async function loadModel( url ) {

	const fetched = await fetch( url );
	const buffer = await fetched.arrayBuffer();

	const model = await fragments.core.load( buffer, {
		modelId: url,
		camera: world.camera.three,
		raw: false,
	} );

	world.scene.three.add( model.object );
	const now = performance.now();
	await fragments.core.update( true );
	const then = performance.now();
	console.log( `Time taken: ${ then - now }ms` );

	return model;

}

const model = await loadModel( '/frags/school_arq.frag' );

const allMeshes = new Group();

const material = new MeshLambertMaterial();

// Add picking meshes (deduplicating geometries to save memory)
const idsWithGeometry = await model.getItemsIdsWithGeometry();
const allMeshesData = await model.getItemsGeometry( idsWithGeometry );

const geometries = new Map();

for ( const itemId in allMeshesData ) {

	const meshData = allMeshesData[ itemId ];
	for ( const geomData of meshData ) {

		if (
			! geomData.positions ||
			! geomData.indices ||
			! geomData.transform ||
			! geomData.representationId
		) {

			continue;

		}

		const representationId = geomData.representationId;
		if ( ! geometries.has( representationId ) ) {

			const geometry = new BufferGeometry();
			geometry.setAttribute( 'position', new Float32BufferAttribute( geomData.positions, 3 ) );
			geometry.setAttribute( 'normal', new Float32BufferAttribute( geomData.normals, 3 ) );
			geometry.setIndex( Array.from( geomData.indices ) );
			geometries.set( representationId, geometry );

		}

		const geometry = geometries.get( representationId );
		const mesh = new Mesh( geometry, material );
		mesh.applyMatrix4( geomData.transform );
		mesh.applyMatrix4( model.object.matrixWorld );
		mesh.updateWorldMatrix( true, true );
		allMeshes.add( mesh );

	}

}

// initialize BVHs
allMeshes.traverse( c => {

	if ( c.geometry && ! c.geometry.boundsTree ) {

		const elCount = c.geometry.index ? c.geometry.index.count : c.geometry.attributes.position.count;
		c.geometry.groups.forEach( group => {

			if ( group.count === Infinity ) {

				group.count = elCount - group.start;

			}

		} );

		c.geometry.boundsTree = new MeshBVH( c.geometry, { maxLeafSize: 1, strategy: SAH } );

	}

} );

// Compute bounding box of allMeshes
allMeshes.updateWorldMatrix( true, true );

const box = new Box3();
allMeshes.traverse( ( child ) => {

	if ( child.isMesh && child.geometry ) {

		child.updateWorldMatrix( false, false );
		box.expandByObject( child, true );

	}

} );

const size = box.getSize( new Vector3() );
const center = box.getCenter( new Vector3() );

const planeHeight = box.max.y + 3;
const planeSize = Math.max( size.x, size.z ) * 1.5;
const planeGeometry = new PlaneGeometry( planeSize, planeSize );
const planeMaterial = new MeshBasicMaterial( {
	color: 0x000000,
	transparent: true,
	opacity: 0.95,
} );
const groundPlane = new Mesh( planeGeometry, planeMaterial );
groundPlane.rotation.x = - Math.PI / 2;
groundPlane.position.set( center.x, planeHeight, center.z );
world.scene.three.add( groundPlane );

const projectionMaterial = new LineBasicMaterial( { color: 0x888888 } );
projection = new LineSegments( new BufferGeometry(), projectionMaterial );
projection.position.y = planeHeight + 0.01;

drawThroughProjection = new LineSegments( new BufferGeometry(), new LineDashedMaterial( { color: 0x444444, dashSize: 0.03, gapSize: 0.03, transparent: true } ) );
drawThroughProjection.position.y = planeHeight + 0.01;
drawThroughProjection.renderOrder = - 1;
world.scene.three.add( projection, drawThroughProjection );

gui = new GUI();
gui.add( params, 'includeIntersectionEdges' );
gui.add( params, 'displayDrawThroughProjection' );
gui.add( params, 'rotate' );
gui.add( params, 'regenerate' );

world.renderer.onBeforeUpdate.add( () => {

	drawThroughProjection.visible = params.displayDrawThroughProjection;

} );

async function updateEdges() {

	outputContainer.innerText = 'Generating...';

	projection.geometry.dispose();
	drawThroughProjection.geometry.dispose();

	projection.geometry = new BufferGeometry();
	drawThroughProjection.geometry = new BufferGeometry();

	const timeStart = window.performance.now();

	const generator = new ProjectionGenerator( gpuRenderer );
	generator.angleThreshold = ANGLE_THRESHOLD;
	generator.includeIntersectionEdges = params.includeIntersectionEdges;

	const input = await new MeshVisibilityCuller( gpuRenderer, { pixelsPerMeter: 0.05 } ).cull( allMeshes );

	const result = await generator.generate( input, {
		onProgress: p => {

			outputContainer.innerText = `Generating... ${ ( p * 100 ).toFixed( 1 ) }%`;

		},
	} );

	drawThroughProjection.geometry.dispose();
	drawThroughProjection.geometry = result.hiddenEdges.getLineGeometry();
	drawThroughProjection.computeLineDistances();

	projection.geometry.dispose();
	projection.geometry = result.visibleEdges.getLineGeometry();
	const trimTime = window.performance.now() - timeStart;

	outputContainer.innerText = `Generation time: ${ trimTime.toFixed( 2 ) }ms`;

}

updateEdges();
