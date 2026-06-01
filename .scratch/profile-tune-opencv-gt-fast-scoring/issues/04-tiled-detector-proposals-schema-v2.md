Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Blocks: .scratch/profile-tune-opencv-gt-fast-scoring/issues/05-fast-profile-selection-scoring-parity.md

# Tiled detector proposals schema v2 (crop-local RLE)

## What to build

Replace **tiled detector proposals** `schema_version` 1 (SAHI object pickles with full-section dense masks) with **schema_version** 2: a list of compact neutral records per `(variant, conf, mask_threshold)` cache. Each record carries **score**, full-image **bbox**, crop-local **COCO RLE**, and **`offset_y` / `offset_x`**; cache metadata records full-section **height** and **width**.

The detector job crops each shifted detection when writing the cache and rejects v1 on load. Roundtrip tests cover write → read → field validation. Held-out `yolo.predict` is unchanged.

## Acceptance criteria

- [x] `TILED_PROPOSAL_CACHE_SCHEMA_VERSION` is 2; v1 caches fail validation with a clear error
- [x] Detector job writes v2 records (no SAHI types on disk)
- [x] Each persisted mask is crop-local RLE + offsets, not a full-section dense plane
- [x] Cache fingerprint sidecar still includes weights hash, recipe window hash, conf, mask_threshold, variant, sample id
- [x] Tests: roundtrip write/load; v1 rejection

## Blocked by

None — can start immediately (parallel with slices 01–03).
