import {
	Box3,
	WebGLRenderer,
	Scene,
	DirectionalLight,
	AmbientLight,
	Group,
	BufferGeometry,
	LineSegments,
	LineBasicMaterial,
	PerspectiveCamera,
	Matrix4,
	Vector3,
} from 'three';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { ProjectionGenerator } from '..';
import { MeshBVH, SAH } from 'three-mesh-bvh';

const params = {
	displayModel: true,
	displayProjection: true,
	displayDrawThrough: false,
	includeIntersectionEdges: false,
	generate: () => {

		task = updateEdges();

	},
};

const ANGLE_THRESHOLD = 50;
let needsRender = false;
let renderer, camera, scene, gui, controls;
let model, projection, drawThroughProjection, group;
let outputContainer;
let task = null;

// module-level temporaries
const _dir = /* @__PURE__ */ new Vector3();
const _box = /* @__PURE__ */ new Box3();
const _VP = /* @__PURE__ */ new Matrix4();
const _R = /* @__PURE__ */ new Matrix4();

init();

async function init() {

	outputContainer = document.getElementById( 'output' );

	const bgColor = 0xeeeeee;

	// renderer setup
	renderer = new WebGLRenderer( { antialias: true } );
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.setClearColor( bgColor, 1 );
	document.body.appendChild( renderer.domElement );

	// scene setup
	scene = new Scene();

	// lights
	const light = new DirectionalLight( 0xffffff, 3.5 );
	light.position.set( 1, 2, 3 );
	scene.add( light );

	const ambientLight = new AmbientLight( 0xb0bec5, 0.5 );
	scene.add( ambientLight );

	group = new Group();
	scene.add( group );

	// load model
	const gltf = await new GLTFLoader()
		.setMeshoptDecoder( MeshoptDecoder )
		.loadAsync( 'https://raw.githubusercontent.com/gkjohnson/3d-demo-data/main/models/nasa-m2020/Perseverance.glb' );
	model = gltf.scene;

	// initialize BVHs
	model.traverse( c => {

		if ( c.geometry && ! c.geometry.boundsTree ) {

			const elCount = c.geometry.index ? c.geometry.index.count : c.geometry.attributes.position.count;
			c.geometry.groups.forEach( g => {

				if ( g.count === Infinity ) g.count = elCount - g.start;

			} );
			c.geometry.boundsTree = new MeshBVH( c.geometry, { maxLeafSize: 1, strategy: SAH } );

		}

	} );

	// center model
	const box = new Box3();
	box.setFromObject( model, true );
	box.getCenter( group.position ).multiplyScalar( - 1 );
	group.add( model );
	group.updateMatrixWorld( true );

	// projection display meshes
	projection = new LineSegments( new BufferGeometry(), new LineBasicMaterial( { color: 0x030303, depthWrite: false } ) );
	drawThroughProjection = new LineSegments( new BufferGeometry(), new LineBasicMaterial( { color: 0xaaaaaa, depthWrite: false } ) );
	drawThroughProjection.renderOrder = - 1;
	scene.add( projection, drawThroughProjection );

	// camera setup
	camera = new PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.01, 1e3 );
	camera.position.set( 3, 2, 3.5 );
	camera.lookAt( 0, 0, 0 );
	camera.updateProjectionMatrix();

	needsRender = true;

	// controls
	controls = new OrbitControls( camera, renderer.domElement );
	controls.addEventListener( 'change', () => {

		needsRender = true;

	} );

	gui = new GUI();
	gui.add( params, 'displayModel' ).onChange( () => needsRender = true );
	gui.add( params, 'displayProjection' ).onChange( () => needsRender = true );
	gui.add( params, 'displayDrawThrough' ).onChange( () => needsRender = true );
	gui.add( params, 'includeIntersectionEdges' );
	gui.add( params, 'generate' );

	render();

	window.addEventListener( 'resize', function () {

		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();

		renderer.setSize( window.innerWidth, window.innerHeight );

		needsRender = true;

	}, false );

}

async function* updateEdges( runTime = 30 ) {

	outputContainer.innerText = 'Generating...';

	// dispose existing geometry
	projection.geometry.dispose();
	drawThroughProjection.geometry.dispose();
	projection.geometry = new BufferGeometry();
	drawThroughProjection.geometry = new BufferGeometry();

	// construct the generation group
	const scaleGroup = new Group();
	const perspectiveGroup = new Group();
	perspectiveGroup.matrixAutoUpdate = false;
	scaleGroup.add( perspectiveGroup );

	const clone = group.clone();
	clone.visible = true;
	perspectiveGroup.add( clone );

	// transform the clone to be relative to the camera
	clone.matrix
		.multiplyMatrices( camera.matrixWorldInverse, group.matrixWorld )
		.decompose( clone.position, clone.quaternion, clone.scale );

	perspectiveGroup.matrix
		.copy( camera.projectionMatrix );

	scaleGroup.scale.z = camera.far * 0.1;
	scaleGroup.rotation.x = Math.PI / 2;


	const b = new Box3();
	b.setFromObject( scaleGroup );
	// scaleGroup.position.z = - b.min.z;
	scaleGroup.updateMatrixWorld( true );
	// console.log( b.min.z )

	// // console.log( camera.near, camera.far )

	// scene.add( scaleGroup );
	needsRender = true;
	// return;



	const timeStart = window.performance.now();

	const generator = new ProjectionGenerator();
	generator.iterationTime = runTime;
	generator.angleThreshold = ANGLE_THRESHOLD;
	generator.includeIntersectionEdges = params.includeIntersectionEdges;

	const collection = yield* generator.generate( [ scaleGroup ], {
		onProgress: ( tot, msg, edges ) => {

			outputContainer.innerText = msg;
			if ( tot ) outputContainer.innerText += ' ' + ( 100 * tot ).toFixed( 1 ) + '%';

			if ( edges ) {

				projection.geometry.dispose();
				projection.geometry = edges.visibleEdges.getLineGeometry();
				needsRender = true;

			}

		},
	} );

	// set final geometries
	projection.geometry.dispose();
	projection.geometry = collection.visibleEdges.getLineGeometry();
	drawThroughProjection.geometry.dispose();
	drawThroughProjection.geometry = collection.hiddenEdges.getLineGeometry();

	const trimTime = window.performance.now() - timeStart;
	outputContainer.innerText = `Generation time: ${ trimTime.toFixed( 2 ) }ms`;

	needsRender = true;

}

function render() {

	requestAnimationFrame( render );

	if ( task ) {

		const res = task.next();
		if ( res.done ) task = null;

	}

	model.visible = params.displayModel;
	projection.visible = params.displayProjection;
	drawThroughProjection.visible = params.displayDrawThrough;

	if ( needsRender ) {

		renderer.render( scene, camera );
		needsRender = false;

	}

}
