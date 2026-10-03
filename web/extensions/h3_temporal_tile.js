import { app } from "../../../scripts/app.js";

// H3TemporalSampler: grey out widgets that have no effect in the current
// state, so they can't be edited by mistake:
//
//   enable = false -> all tiling widgets off
//
// Works in the classic canvas and Nodes 2.0 (Vue) frontends: Vue also needs
// widget.options.disabled, and saved workflows apply widget values AFTER
// nodeCreated, so the rules are re-applied on loadedGraphNode.

function setDisabled(w, d) {
	w.disabled = d;
	try {
		if (w.options) w.options.disabled = d;
	} catch (_) {}
}

function wire(node) {
	if (!node || node.comfyClass !== "H3TemporalSampler") return;
	const W = {};
	for (const w of (node.widgets || [])) W[w.name] = w;
	const names = ["enable", "step_average", "num_segments", "smart_bounds",
	               "overlap_frames", "blend_mode"];
	if (!names.every((n) => W[n])) return;

	const sync = () => {
		const tilingOff = !W["enable"].value;
		const stepAvg = !!W["step_average"].value;
		setDisabled(W["step_average"], tilingOff);
		setDisabled(W["num_segments"], tilingOff);
		setDisabled(W["smart_bounds"], tilingOff);
		setDisabled(W["overlap_frames"], tilingOff);
		// blend_mode only matters in the legacy per-segment path;
		// step_average does its own cosine consensus during sampling.
		setDisabled(W["blend_mode"], tilingOff || stepAvg);
		try { node.setDirtyCanvas(true, true); } catch (_) {}
	};

	for (const key of ["enable", "step_average"]) {
		const w = W[key];
		if (!w._h3t_wired) {
			w._h3t_wired = true;
			const orig = w.callback;
			w.callback = function (...args) {
				if (orig) orig.apply(this, args);
				sync();
			};
		}
	}
	sync();
}

app.registerExtension({
	name: "h3_temporal_tile.widgets",
	nodeCreated(node) { wire(node); },
	loadedGraphNode(node) { wire(node); },
});
