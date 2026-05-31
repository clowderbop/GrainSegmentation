# YOLO score merge at predict time

Status: accepted

ADR 0001 made the **instance prediction set** canonical but persisted YOLO output as overlapping **detector proposals**, building a transient **merged instance view** only at evaluation. That split made SAHI **prediction overlays** look like split or duplicated grains when proposals did not overlap (e.g. **slice-boundary duplicate**), while headline AJI/F1 already scored the merged layout — the same JSON file did not represent the YOLO **system** we compare to U-Net.

We now treat **score merge** as part of the YOLO system: `yolo.predict` (whole SAHI and patch) merges proposals, then writes only non-overlapping grains to `prediction_sets/{sample_id}.json`, each with the **score** of its winning proposal. **Mask AP** runs standard COCO instance mask AP on that canonical set (not on raw proposals). Eval, overlays, and reporting read the same artifact; `evaluate_instances` no longer merges YOLO sets at metric time. **Slice-boundary duplicate** remains an accepted limitation — score merge does not fuse non-overlapping halves of one grain.

**Considered options:** Merge only at eval/viz (rejected — disk artifact ≠ system output); dual-write proposals + merged (rejected — complexity); merge only on whole-section runs (rejected — patch and whole must share one schema); boundary fusion after merge (deferred — separate decision).

**Consequences:** Supersedes the YOLO persistence and mask-AP input story in ADR 0001. Hard break: delete pre-change `eval/yolo_*` and `eval/yolo_patches/*` on scratch and re-run default YOLO test jobs before updating **post-eval reporting**. Ultralytics patch val mAP stays on native detector output (**supporting test metrics**). Update `CONTEXT.md` (done in grill session). Implementation tracked in `.scratch/yolo-score-merge-at-predict/`.
