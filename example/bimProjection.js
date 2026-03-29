import * as THREE from 'three';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import * as OBC from '@thatopen/components';
import * as WEBIFC from 'web-ifc';
import { GeometryEngine } from '@thatopen/fragments';
import { ProjectionGenerator, VisibilityCuller, PlanarIntersectionGenerator } from '..';
import {Logger} from '../src/utils/Logger.js';


const params = {
	displayModel: true,
	logging: true,
	displayDrawThroughProjection: false,
	includeIntersectionEdges: false,
	useWebGPU: true,
	enableClipping: false,
	displayClippingEdges: true,
	rotate: () => {

		const randomQuaternion = new THREE.Quaternion();
		randomQuaternion.random();

		allMeshes.quaternion.copy(randomQuaternion);
		allMeshes.position.set(0, 0, 0);
		allMeshes.updateMatrixWorld(true);

	},
	regenerate: () => {

		task = updateEdges();

	},
};

const ANGLE_THRESHOLD = 50;
let gui;
let projection, drawThroughProjection;
let outputContainer;
let task = null;
Logger.enabled = params.logging;


const components = new OBC.Components();
const worlds = components.get(OBC.Worlds);
const container = document.getElementById("container");

const world = worlds.create();

world.scene = new OBC.SimpleScene(components);
world.renderer = new OBC.SimpleRenderer(components, container);
world.camera = new OBC.OrthoPerspectiveCamera(components);

components.init();

world.scene.setup();

world.scene.three.add(new THREE.AxesHelper());

outputContainer = document.getElementById('output');


// Initialize GeometryEngine for boolean operations
const ifcApi = new WEBIFC.IfcAPI();
ifcApi.SetWasmPath('https://unpkg.com/web-ifc@0.0.75/', false);
await ifcApi.Init();
const geometryEngine = new GeometryEngine(ifcApi);


// load model

// prettier-ignore
const githubUrl =
	"https://thatopen.github.io/engine_fragment/resources/worker.mjs";
const fetchedUrl = await fetch(githubUrl);
const workerBlob = await fetchedUrl.blob();
const workerFile = new File([workerBlob], "worker.mjs", {
	type: "text/javascript",
});
const workerUrl = URL.createObjectURL(workerFile);
const fragments = components.get(OBC.FragmentsManager);
fragments.init(workerUrl);

world.camera.controls.addEventListener("control", () =>
	fragments.core.update(true),
);

// Remove z fighting
fragments.core.models.materials.list.onItemSet.add(({ value: material }) => {
	if (!("isLodMaterial" in material && material.isLodMaterial)) {
		material.polygonOffset = true;
		material.polygonOffsetUnits = 1;
		material.polygonOffsetFactor = Math.random();
	}
});

async function loadModel(
	url,
	id = url,
	raw = false,
) {
	const fetched = await fetch(url);
	const buffer = await fetched.arrayBuffer();

	const model = await fragments.core.load(buffer, {
		modelId: id,
		camera: world.camera.three,
		raw,
	});

	world.scene.three.add(model.object);
	// model.object.rotation.x = Math.PI / 4;
	// model.object.rotation.y = Math.PI / 4;
	const now = performance.now();
	await fragments.core.update(true);
	const then = performance.now();
	console.log(`Time taken: ${then - now}ms`);

	return model;
}

const model = await loadModel("/frags/m3d.frag");

const allMeshes = new THREE.Group();
// world.scene.three.add(allMeshes);

// Separate group for clipped results
const clippedMeshes = new THREE.Group();
// world.scene.three.add(clippedMeshes);

const material = new THREE.MeshLambertMaterial({
	color: new THREE.Color("white"),
});

// Add picking meshes (deduplicating geometries to save memory)
const idsWithGeometry = await model.getItemsIdsWithGeometry();
const allMeshesData = await model.getItemsGeometry(idsWithGeometry);

const geometries = new Map();

for (const itemId in allMeshesData) {

	const meshData = allMeshesData[itemId];
	for (const geomData of meshData) {
		if (
			!geomData.positions ||
			!geomData.indices ||
			!geomData.transform ||
			!geomData.representationId
		) {
			continue;
		}

		const representationId = geomData.representationId;
		if (!geometries.has(representationId)) {
			const geometry = new THREE.BufferGeometry();
			geometry.setAttribute(
				"position",
				new THREE.Float32BufferAttribute(geomData.positions, 3),
			);
			geometry.setAttribute(
				"normal",
				new THREE.Float32BufferAttribute(geomData.normals, 3),
			);
			geometry.setIndex(Array.from(geomData.indices));
			geometries.set(representationId, geometry);
		}

		const geometry = geometries.get(representationId);

		const mesh = new THREE.Mesh(geometry, material);

		mesh.applyMatrix4(geomData.transform);
		mesh.applyMatrix4(model.object.matrixWorld);
		mesh.updateWorldMatrix(true, true);
		allMeshes.add(mesh);
	}
}

// initialize BVHs
allMeshes.traverse(c => {

	if (c.geometry && !c.geometry.boundsTree) {

		const elCount = c.geometry.index ? c.geometry.index.count : c.geometry.attributes.position.count;
		c.geometry.groups.forEach(group => {

			if (group.count === Infinity) {

				group.count = elCount - group.start;

			}

		});

		c.geometry.boundsTree = new MeshBVH(c.geometry, { maxLeafSize: 1, strategy: SAH });

	}

});

// Compute bounding box of allMeshes
allMeshes.updateWorldMatrix(true, true);
const box = new THREE.Box3();
allMeshes.traverse((child) => {
	if (child.isMesh && child.geometry) {
		child.updateWorldMatrix(false, false);
		box.expandByObject(child, true);
	}
});

const size = box.getSize(new THREE.Vector3());
const center = box.getCenter(new THREE.Vector3());

console.log('Model bounds:', box.min.toArray(), box.max.toArray());
console.log('Model size:', size.toArray(), 'center:', center.toArray());

// Create white ground plane on top of the bounding box (plus 3m offset)
const planeHeight = box.max.y + 3;
const planeSize = Math.max(size.x, size.z) * 1.5;
const planeGeometry = new THREE.PlaneGeometry(planeSize, planeSize);
const planeMaterial = new THREE.MeshBasicMaterial({
	color: 0xffffff,
	transparent: true,
	opacity: 0.95,
});
const groundPlane = new THREE.Mesh(planeGeometry, planeMaterial);
groundPlane.rotation.x = -Math.PI / 2; // Rotate to be horizontal
groundPlane.position.set(center.x, planeHeight, center.z);
world.scene.three.add(groundPlane);

const clipper = components.get(OBC.Clipper);
// const clipNormal = new THREE.Vector3(0, 1, 0).applyEuler(new THREE.Euler(Math.PI / 2, 0, 0)).applyEuler(new THREE.Euler(Math.PI / 4, Math.PI / 4, 0));
const planeId = clipper.createFromNormalAndCoplanarPoint(world, new THREE.Vector3(0, 1, 0), new THREE.Vector3(-50, 50, 0))
const plane = clipper.list.get(planeId);

// --- Clipping edge projection ---
const clippingEdgeMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });
const clippingEdgesGroup = new THREE.Group();
clippingEdgesGroup.position.y = planeHeight + 0.02;
world.scene.three.add(clippingEdgesGroup);

const intersectingMeshes = new Set();

function generateClippingEdges() {

	// Clear previous clipping edges
	for (const child of [...clippingEdgesGroup.children]) {

		clippingEdgesGroup.remove(child);
		if (child.geometry) child.geometry.dispose();

	}

	const clipPlane = plane.three;
	const generator = new PlanarIntersectionGenerator();
	const invMatrix = new THREE.Matrix4();
	const v = new THREE.Vector3();

	// Ensure world matrices are up to date
	allMeshes.updateWorldMatrix(true, true);

	let totalSegments = 0;
	intersectingMeshes.clear();

	for (const child of allMeshes.children) {

		if (!child.isMesh || !child.geometry) continue;

		// Transform clip plane to mesh's local space
		invMatrix.copy(child.matrixWorld).invert();
		const localPlane = clipPlane.clone().applyMatrix4(invMatrix);

		generator.plane.copy(localPlane);

		const bvh = child.geometry.boundsTree || child.geometry;
		const edgeGeom = generator.generate(bvh);

		const posAttr = edgeGeom.getAttribute('position');
		if (!posAttr || posAttr.count === 0) continue;

		// This mesh actually has triangles crossing the plane
		intersectingMeshes.add(child);

		// Transform positions back to world space and project (flatten Y)
		const positions = posAttr.array;
		for (let i = 0; i < positions.length; i += 3) {

			v.set(positions[i], positions[i + 1], positions[i + 2]);
			v.applyMatrix4(child.matrixWorld);
			positions[i] = v.x;
			positions[i + 1] = 0;
			positions[i + 2] = v.z;

		}

		posAttr.needsUpdate = true;

		const line = new THREE.LineSegments(edgeGeom, clippingEdgeMaterial);
		clippingEdgesGroup.add(line);
		totalSegments += posAttr.count / 2;

	}

	console.log(`Clipping edges: ${totalSegments} line segments from ${clippingEdgesGroup.children.length} meshes (${intersectingMeshes.size} intersecting)`);

}


// --- Boolean clipping ---
function applyClipping() {
	// Clear previous clipped meshes
	const previous = [...clippedMeshes.children];
	for (const child of previous) {
		clippedMeshes.remove(child);
		if (child.geometry) child.geometry.dispose();
	}

	const clipPlane = plane.three;
	const n = clipPlane.normal;

	// Create clipping box: a large box on the clipped side (above the plane)
	const boxSize = Math.max(size.x, size.y, size.z) * 4;
	const clipBoxGeom = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
	const clipBoxMesh = new THREE.Mesh(clipBoxGeom, material);
	// Orient and position the box on the negative side of the clip plane
	clipBoxMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), n);
	const coplanarPoint = new THREE.Vector3();
	clipPlane.coplanarPoint(coplanarPoint);
	clipBoxMesh.position.copy(coplanarPoint).addScaledVector(n, -boxSize / 2);
	clipBoxMesh.updateMatrixWorld(true);

	let clipped = 0, skipped = 0, errors = 0, kept = 0;

	for (const child of allMeshes.children) {
		if (!child.isMesh || !child.geometry) continue;

		// Use the precise intersection test from generateClippingEdges()
		if (!intersectingMeshes.has(child)) {
			// No triangles cross the plane — check which side mesh center is on
			const meshCenter = new THREE.Box3().setFromObject(child, true).getCenter(new THREE.Vector3());
			const dist = clipPlane.distanceToPoint(meshCenter);

			if (dist >= 0) {
				// Positive side — keep as-is
				const keepMesh = new THREE.Mesh(child.geometry.clone(), material);
				keepMesh.applyMatrix4(child.matrixWorld);
				keepMesh.updateMatrixWorld(true);
				clippedMeshes.add(keepMesh);
				kept++;
			} else {
				skipped++;
			}
			continue;
		}

		// Actually straddles the clip plane — boolean DIFFERENCE
		try {
			child.updateMatrixWorld(true);

			const booleanData = {
				type: "DIFFERENCE",
				target: child,
				operands: [clipBoxMesh],
			};

			const resultGeom = new THREE.BufferGeometry();
			geometryEngine.getBooleanOperation(resultGeom, booleanData);

			// Check if result has vertices
			const posAttr = resultGeom.getAttribute('position');
			if (!posAttr || posAttr.count === 0) {
				skipped++;
				continue;
			}

			// Result is in world space, so create mesh with identity transform
			const resultMesh = new THREE.Mesh(resultGeom, material);
			resultMesh.updateMatrixWorld(true);
			clippedMeshes.add(resultMesh);
			clipped++;
		} catch (e) {
			console.warn('Boolean op error:', e);
			// On error, keep the original mesh
			const fallbackMesh = new THREE.Mesh(child.geometry.clone(), material);
			fallbackMesh.applyMatrix4(child.matrixWorld);
			fallbackMesh.updateMatrixWorld(true);
			clippedMeshes.add(fallbackMesh);
			errors++;
		}
	}

	console.log(`Boolean clipping: clipped=${clipped}, kept=${kept}, skipped=${skipped}, errors=${errors}, total=${allMeshes.children.length}`);

	// Build BVHs for clipped meshes
	clippedMeshes.traverse(c => {
		if (c.geometry && !c.geometry.boundsTree) {
			const elCount = c.geometry.index ? c.geometry.index.count : c.geometry.attributes.position.count;
			c.geometry.groups.forEach(group => {
				if (group.count === Infinity) {
					group.count = elCount - group.start;
				}
			});
			c.geometry.boundsTree = new MeshBVH(c.geometry, { maxLeafSize: 1, strategy: SAH });
		}
	});

	// Hide original meshes, show clipped
	// allMeshes.visible = false;
	// model.object.visible = false;
	// clippedMeshes.visible = true;
}



// create projection display mesh
const projectionMaterial = new THREE.LineBasicMaterial({ color: 0x888888 });
projection = new THREE.LineSegments(new THREE.BufferGeometry(), projectionMaterial);
projection.position.y = planeHeight + 0.01;

drawThroughProjection = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0x444444, dashSize: 0.03, gapSize: 0.03, transparent: true }));
drawThroughProjection.position.y = planeHeight + 0.01;
drawThroughProjection.renderOrder = - 1;
world.scene.three.add(projection, drawThroughProjection);

gui = new GUI();
gui.add(params, 'includeIntersectionEdges');
gui.add(params, 'useWebGPU');
gui.add(params, 'displayDrawThroughProjection');
gui.add(params, 'enableClipping');
gui.add(params, 'displayClippingEdges');
gui.add(params, 'logging').onChange(() => {
	Logger.enabled = params.logging;
});
gui.add(params, 'rotate');
gui.add(params, 'regenerate');

world.renderer.onBeforeUpdate.add(() => {

	if (task) {

		const res = task.next();
		if (res.done) {

			task = null;

		}

	}

	drawThroughProjection.visible = params.displayDrawThroughProjection;
	clippingEdgesGroup.visible = params.displayClippingEdges;

});


function* updateEdges(runTime = 30) {

	outputContainer.innerText = 'Generating...';

	// dispose the geometry
	projection.geometry.dispose();
	drawThroughProjection.geometry.dispose();

	// initialize an empty geometry
	projection.geometry = new THREE.BufferGeometry();
	drawThroughProjection.geometry = new THREE.BufferGeometry();

	const timeStart = window.performance.now();

	if (params.enableClipping) {
		generateClippingEdges();
		applyClipping();
	}

	const generator = new ProjectionGenerator();
	generator.iterationTime = runTime;
	generator.angleThreshold = ANGLE_THRESHOLD;
	generator.includeIntersectionEdges = params.includeIntersectionEdges;
	generator.useWebGPU = params.useWebGPU;
	console.log(generator.includeIntersectionEdges);

	// Use clippedMeshes if clipping is enabled, otherwise allMeshes
	const meshSource = params.enableClipping ? clippedMeshes : allMeshes;

	const culler = new VisibilityCuller(world.renderer.three, { pixelsPerMeter: 0.01 });
	culler.clippingPlanes = [plane.three];

	const collection = yield* generator.generate(meshSource, {
		visibilityCuller: culler,
		onProgress: (msg, tot, edges) => {

			outputContainer.innerText = msg;
			if (tot) outputContainer.innerText += ' ' + (100 * tot).toFixed(1) + '%';

			if (edges) {

				projection.geometry.dispose();
				projection.geometry = edges.getVisibleLineGeometry();

			}

		},
	});
	drawThroughProjection.geometry.dispose();
	drawThroughProjection.geometry = collection.getHiddenLineGeometry();
	drawThroughProjection.computeLineDistances();

	projection.geometry.dispose();
	projection.geometry = collection.getVisibleLineGeometry();
	const trimTime = window.performance.now() - timeStart;

	outputContainer.innerText = `Generation time: ${trimTime.toFixed(2)}ms`;

}

// task = updateEdges();
