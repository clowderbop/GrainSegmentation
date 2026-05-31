Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: —
Blocks: —

# Implement YOLO score merge at predict time

## Parent

[PRD: YOLO score merge at predict](../PRD.md) · [ADR-0004](../../../docs/adr/0004-yolo-score-merge-at-predict.md)

## What to build

Move **score merge** from eval-only into `yolo.predict` for both `unit=whole` (SAHI) and `unit=patch` (Ultralytics per crop).

1. **Merge + export** — After collecting **detector proposals** (in memory), run existing merge (`yolo_detections_to_instance_map_by_score` or equivalent). Convert the label map to a **instance prediction set**: one RLE per label id with any pixels; **score** = the winning proposal’s score; drop empty masks. Write `prediction_sets/{sample_id}.json` with `producer: yolo` (non-overlapping grains only).

2. **Provenance** — Extend **run provenance** to record that score merge ran at predict (e.g. `score_merge_at_predict: true`); keep existing SAHI/conf fields.

3. **Eval simplification** — `prediction_set_to_merged_instance_view` for `producer: yolo` should rasterize the canonical set (or assert non-overlap and assign labels 1..N). Remove redundant merge-at-eval if redundant.

4. **Mask AP** — No code path that assumes overlapping YOLO proposals; `yolo_prediction_set_to_coco_dt` unchanged in spirit (one dt per grain + score).

5. **Visualization** — `export_sahi_visualization` draws canonical grains (no score-order overlap resolution needed unless kept for identical colours); must match merged geometry.

6. **Validation** — Optional: assert or document that YOLO prediction sets have pairwise non-overlapping masks after predict.

7. **Tests** — Unit tests: proposals with overlap → merged JSON has one mask per surviving label, correct scores; patch path calls same export; merged view from saved set matches pre-change merge behaviour for a fixture.

8. **Docs** — Update `SLURM/yolo/pipeline.md` (prediction set = post-merge grains; re-run required). Touch ADR 0003 supporting-metrics sentence only if it still says “proposals” for Mask AP.

## Operator steps (not code)

- Delete pre-change `eval/yolo_{variant}/` and `eval/yolo_patches/{variant}/*/` on scratch.
- Re-run `bash SLURM/yolo/submit_test_evaluations.sh`.
- Regenerate **post-eval reporting** bundle.

## Acceptance criteria

- [ ] Whole SAHI predict writes non-overlapping `prediction_sets/*.json`; overlapping proposal fixture merges to expected RLEs and scores
- [ ] Patch predict uses the same merge+export path
- [ ] `evaluate_instances` on YOLO whole/patch uses canonical set without double-merge
- [ ] `evaluate_mask_ap` runs on canonical sets; tests updated if fixtures assumed overlapping proposals
- [ ] SAHI overlay export reflects canonical grains (split-boundary duplicates may remain — not fixed here)
- [ ] Run provenance documents score merge at predict
- [ ] Pipeline doc notes hard break and re-submit

## Comments

- Grill session 2026-05-31: canonical output = non-overlapping grains (A); winning proposal score for COCO (B); hard break on scratch eval (A); **slice-boundary duplicate** accepted (A).
