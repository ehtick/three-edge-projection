<!-- This file is generated automatically. Do not edit it directly. -->
# three-edge-projection

## Constants

### OUTPUT_MESH

```js
OUTPUT_MESH: number
```

### OUTPUT_LINE_SEGMENTS

```js
OUTPUT_LINE_SEGMENTS: number
```

### OUTPUT_BOTH

```js
OUTPUT_BOTH: number
```

## EdgeSet

Set of projected edges produced by ProjectionGenerator.


### .getLineGeometry

```js
getLineGeometry( meshes = null: Array<Mesh> | null ): BufferGeometry
```

Returns a new BufferGeometry representing the edges.

Pass a list of meshes in to extract edges from a specific subset of meshes in the given
order. Returns all edges if null.


### .getRangeForMesh

```js
getRangeForMesh( mesh: Mesh ): Object | null
```

Returns the range of vertices associated with the given mesh in the geometry returned from
getLineGeometry. The `start` value is only relevant if lines are generated with the default
order and set of meshes.

Can be used to add extra vertex attributes in a geometry associated with a specific subrange
of the geometry.


## MeshVisibilityCuller

Utility for determining visible geometry from a top down orthographic perspective. This can
be run before performing projection generation to reduce the complexity of the operation at
the cost of potentially missing small details.

Constructor for the visibility culler that takes the renderer to use for culling.


### .pixelsPerMeter

```js
pixelsPerMeter: number
```

The size of a pixel on a single dimension. If this results in a texture larger than what
the graphics context can provide then the rendering is tiled.


### .constructor

```js
constructor(
	renderer: WebGLRenderer,
	{
		pixelsPerMeter = 0.1: number,
	}
)
```

### .cull

```js
async cull( object: Object3D | Array<Object3D> ): Promise<Array<Object3D>>
```

Returns the set of meshes that are visible within the given object.


## PlanarIntersectionGenerator

Utility for generating the line segments produced by a planar intersection with geometry.


### .plane

```js
plane: Plane
```

Plane that defaults to y up plane at the origin.


### .generate

```js
generate( geometry: MeshBVH | BufferGeometry ): BufferGeometry
```

Generates a geometry of the resulting line segments from the planar intersection.


## ProjectionGenerator

Utility for generating 2D projections of 3D geometry.


### .iterationTime

```js
iterationTime: number
```

How long to spend trimming edges before yielding.


### .angleThreshold

```js
angleThreshold: number
```

The threshold angle in degrees at which edges are generated.


### .includeIntersectionEdges

```js
includeIntersectionEdges: boolean
```

Whether to generate edges representing the intersections between triangles.


### .generateAsync

```js
async generateAsync(
	geometry: Object3D | BufferGeometry | Array<Object3D>,
	{
		onProgress?: (
			percent: number,
			message: string
		) => void,
		signal?: AbortSignal,
	}
): ProjectionResult
```

Generate the geometry with a promise-style API.


### .generate

```js
generate(
	scene: Object3D | BufferGeometry | Array<Object3D>,
	{
		onProgress?: (
			percent: number,
			message: string
		) => void,
	}
): ProjectionResult
```

Generate the edge geometry result using a generator function.


## ProjectionResult

Result object returned by ProjectionGenerator containing visible and hidden edge sets.


### .visibleEdges

```js
visibleEdges: EdgeSet
```


### .hiddenEdges

```js
hiddenEdges: EdgeSet
```


## SilhouetteGenerator

Used for generating a projected silhouette of a geometry using the clipper2-js project. Performing
these operations can be extremely slow with more complex geometry and not always yield a stable result.


### .iterationTime

```js
iterationTime: number
```

How long to spend trimming edges before yielding.


### .doubleSided

```js
doubleSided: boolean
```

If `false` then only the triangles facing upwards are included in the silhouette.


### .sortTriangles

```js
sortTriangles: boolean
```

Whether to sort triangles and project them large-to-small. In some cases this can cause
the performance to drop since the union operation is best performed with smooth, simple
edge shapes.


### .output

```js
output: number
```

Whether to output mesh geometry, line segments geometry, or both in an array
( `[ mesh, line segments ]` ).


### .generateAsync

```js
async generateAsync(
	geometry: BufferGeometry,
	{
		onProgress?: (
			percent: number
		) => void,
		signal?: AbortSignal,
	}
): BufferGeometry | Array<BufferGeometry>
```

Generate the silhouette geometry with a promise-style API.


### .generate

```js
generate(
	geometry: BufferGeometry,
	{
		onProgress?: (
			percent: number
		) => void,
	}
): BufferGeometry | Array<BufferGeometry>
```

Generate the geometry using a generator function.


# WebGPU API

## MeshVisibilityCuller

Utility for determining visible geometry from a top down orthographic perspective. This can
be run before performing projection generation to reduce the complexity of the operation at
the cost of potentially missing small details.

Takes the WebGPURenderer instance used to render.


### .pixelsPerMeter

```js
pixelsPerMeter: number
```

The size of a pixel on a single dimension. If this results in a texture larger than what
the graphics context can provide then the rendering is tiled.


### .constructor

```js
constructor(
	renderer: WebGPURenderer,
	{
		pixelsPerMeter = 0.1: number,
	}
)
```

### .cull

```js
async cull( object: Object3D | Array<Object3D> ): Promise<Array<Object3D>>
```

Returns the set of meshes that are visible within the given object.


## ProjectionGenerator

Takes the WebGPURenderer instance used to run compute kernels.


### .angleThreshold

```js
angleThreshold: number
```

The threshold angle in degrees at which edges are generated.


### .batchSize

```js
batchSize: number
```

The number of edges to process in one compute kernel pass. Larger values can process
faster but may cause internal buffers to overflow, resulting in extra kernel executions,
taking more time.


### .includeIntersectionEdges

```js
includeIntersectionEdges: boolean
```

Whether to generate edges representing the intersections between triangles.


### .iterationTime

```js
iterationTime: number
```

How long to spend generating edges.


### .constructor

```js
constructor( renderer: WebGPURenderer )
```

### .generate

```js
async generate(
	scene: Object3D | BufferGeometry | Array<Object3D>,
	{
		onProgress?: (
			percent: number,
			message: string
		) => void,
		signal?: AbortSignal,
	}
): Promise<ProjectionResult>
```

Asynchronously generate the edge geometry result.

