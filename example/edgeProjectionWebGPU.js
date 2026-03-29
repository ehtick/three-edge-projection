import {
	Box3,
	Scene,
	DirectionalLight,
	AmbientLight,
	Group,
	BufferGeometry,
	LineSegments,
	LineBasicMaterial,
	PerspectiveCamera,
	WebGPURenderer,
} from 'three/webgpu';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import { ComputeProjectionGenerator, MeshVisibilityCuller } from 'three-edge-projection/webgpu';

const params = {
	displayModel: true,
	displayDrawThroughProjection: false,
	includeIntersectionEdges: false,
	visibilityCullMeshes: false,
	regenerate: () => {

		updateEdges();

	},
	rotate: () => {

		group.quaternion.random();
		group.position.set( 0, 0, 0 );
		group.updateMatrixWorld( true );

		const box = new Box3();
		box.setFromObject( model, true );
		box.getCenter( group.position ).multiplyScalar( - 1 );
		group.position.y = Math.max( 0, - box.min.y ) + 1;
		group.updateMatrixWorld( true );

		needsRender = true;

	},
};

let needsRender = false;
let renderer, camera, scene, gui, controls;
let model, projection, drawThroughProjection, group;
let outputContainer;

init();

async function init() {

	outputContainer = document.getElementById( 'output' );

	const bgColor = 0xeeeeee;

	// renderer setup
	renderer = new WebGPURenderer( { antialias: true } );
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.setClearColor( bgColor, 1 );
	await renderer.init();
	document.body.appendChild( renderer.domElement );

	// scene setup
	scene = new Scene();

	// lights
	const light = new DirectionalLight( 0xffffff, 3.5 );
	light.position.set( 1, 2, 3 );
	scene.add( light );

	const ambientLight = new AmbientLight( 0xb0bec5, 0.5 );
	scene.add( ambientLight );

	// load model
	group = new Group();
	window.GROUP = group;
	group.rotation.set( 0.3, 0, 0.3 );
	group.updateMatrixWorld();
	scene.add( group );

	const gltf = await new GLTFLoader()
		.setMeshoptDecoder( MeshoptDecoder )
		// .loadAsync( 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/nasa-m2020/Perseverance.glb' );
		.loadAsync( new URL( './simple.glb', import.meta.url ).toString() );
	model = gltf.scene;

	// initialize BVHs
	model.traverse( c => {

		if ( c.geometry && ! c.geometry.boundsTree ) {

			const elCount = c.geometry.index ? c.geometry.index.count : c.geometry.attributes.position.count;
			c.geometry.groups.forEach( grp => {

				if ( grp.count === Infinity ) {

					grp.count = elCount - grp.start;

				}

			} );

			c.geometry.boundsTree = new MeshBVH( c.geometry, { maxLeafSize: 1, strategy: SAH } );

		}

	} );

	// center model
	const box = new Box3();
	box.setFromObject( model, true );
	box.getCenter( group.position ).multiplyScalar( - 1 );
	group.position.y = Math.max( 0, - box.min.y ) + 1;
	group.add( model );
	group.updateMatrixWorld( true );

	// create projection display meshes
	projection = new LineSegments( new BufferGeometry(), new LineBasicMaterial( { color: 0x030303, depthWrite: false } ) );
	drawThroughProjection = new LineSegments( new BufferGeometry(), new LineBasicMaterial( { color: 0xcacaca, depthWrite: false } ) );
	drawThroughProjection.renderOrder = - 1;
	scene.add( projection, drawThroughProjection );

	// camera setup
	camera = new PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.01, 1e6 );
	camera.position.setScalar( 100.5 );
	camera.updateProjectionMatrix();

	needsRender = true;

	// controls
	controls = new OrbitControls( camera, renderer.domElement );
	controls.addEventListener( 'change', () => {

		needsRender = true;

	} );

	gui = new GUI();
	gui.add( params, 'displayModel' ).onChange( () => needsRender = true );
	gui.add( params, 'displayDrawThroughProjection' ).onChange( () => needsRender = true );
	gui.add( params, 'includeIntersectionEdges' );
	gui.add( params, 'visibilityCullMeshes' );
	gui.add( params, 'rotate' );
	gui.add( params, 'regenerate' );

	render();

	updateEdges();

	window.addEventListener( 'resize', function () {

		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();

		renderer.setSize( window.innerWidth, window.innerHeight );

		needsRender = true;

	}, false );

}

async function updateEdges() {

	outputContainer.innerText = 'Generating...';

	projection.geometry.dispose();
	projection.material.dispose();
	projection.geometry = new BufferGeometry();

	drawThroughProjection.geometry.dispose();
	drawThroughProjection.material.dispose();
	drawThroughProjection.geometry = new BufferGeometry();

	needsRender = true;

	const timeStart = window.performance.now();
	const generator = new ComputeProjectionGenerator( renderer );
	generator.includeIntersectionEdges = params.includeIntersectionEdges;

	let input = [ model ];
	if ( params.visibilityCullMeshes ) {

		input = await new MeshVisibilityCuller( renderer, { pixelsPerMeter: 0.1 } ).cull( input );

	}

	const result = await generator.generate( input, {
		onProgress: p => {

			outputContainer.innerText = `Generating... ${ ( p * 100 ).toFixed( 2 ) }%`;

		},
	} );
	projection.geometry.dispose();
	projection.material.dispose();
	projection.geometry = result.visibleEdges.getLineGeometry();

	drawThroughProjection.geometry.dispose();
	drawThroughProjection.geometry = result.hiddenEdges.getLineGeometry();

	const elapsed = window.performance.now() - timeStart;
	outputContainer.innerText = `Generation time: ${ elapsed.toFixed( 2 ) }ms`;

	needsRender = true;

}

function render() {

	requestAnimationFrame( render );

	model.visible = params.displayModel;
	drawThroughProjection.visible = params.displayDrawThroughProjection;

	if ( needsRender ) {

		renderer.render( scene, camera );
		needsRender = false;

	}

}
