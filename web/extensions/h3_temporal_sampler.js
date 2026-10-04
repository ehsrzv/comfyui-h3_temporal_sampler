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
	const names = ["Enable", "Num Segments", "Smart Bounds",
	               "Overlap Frames", "Blend Mode", "Seam Lock",
	               "Seam Lock Frames"];
	if (!names.every((n) => W[n])) return;

	const sync = () => {
		const tilingOff = !W["Enable"].value;
		setDisabled(W["Num Segments"], tilingOff);
		setDisabled(W["Smart Bounds"], tilingOff);
		setDisabled(W["Overlap Frames"], tilingOff);
		setDisabled(W["Blend Mode"], tilingOff);
		setDisabled(W["Seam Lock"], tilingOff);
		setDisabled(W["Seam Lock Frames"], tilingOff || !W["Seam Lock"].value);
		try { node.setDirtyCanvas(true, true); } catch (_) {}
	};

	const w = W["Enable"];
	const wl = W["Seam Lock"];
	for (const ww of [w, wl]) {
		if (!ww._h3t_wired) {
			ww._h3t_wired = true;
			const orig = ww.callback;
			ww.callback = function (...args) {
				if (orig) orig.apply(this, args);
				sync();
			};
		}
	}
	sync();
}

app.registerExtension({
	name: "h3_temporal_sampler.widgets",
	nodeCreated(node) { wire(node); },
	loadedGraphNode(node) { wire(node); },
});
