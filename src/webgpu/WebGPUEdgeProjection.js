export class WebGPUEdgeProjection {

	// strategy
	// 1. run a first pass counting the total number of edges required
	// 2. run a second pass accounting for the fact that we know how many edges are needed and early out if needed
	// planning to tackle that edge in the next pass.
	//   - generate pre-merged, sorted edges, compacted edges in per-invocation arrays to limit the number of unnecessary overlays?
	//   - can "flush" arrays once full into the final output buffer to be able to continue
	// 3. copy overlaps to the CPU
	// 4. sort and merge edges
	//   - can limit the edges processed to those completed only?
	// 5. generate final output

}
