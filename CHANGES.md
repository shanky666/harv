# Changes made — performance fix + a broken-page bug found along the way

## 1. Real bug: scan.html was silently broken (not in your original diagnosis, found while fixing the polling)
`frontend/scan.html` — `render()`'s big template literal (`innerHTML = \`...`) was
never closed with a matching backtick before the function's closing brace. That's
a hard JavaScript syntax error: the entire inline `<script>` block fails to parse
in any browser, so `load()` never runs, and the page is stuck forever on the
static "Loading scan results..." markup in the HTML — regardless of how fast or
slow the backend is. This alone could fully explain "the scan result page never
finishes" independent of the performance issues below. Fixed by adding the
missing closing backtick + semicolon.

## 2. Redundant detection pass
`backend/backend/app/services/yolo_service.py` — `detect_fruits()` used to always
run both the OpenCV watershed segmentation pass AND real YOLOv8 inference, then
merge the results. Now it only falls back to the watershed pass when no real
model is loaded.

## 3. Non-blocking analysis (biggest perceived-speed fix)
- `backend/backend/app/api/analyze_basket.py` — `/analyze-basket` now saves the
  upload, schedules the full pipeline as a FastAPI `BackgroundTasks` job, and
  returns `{session_id, status: "processing"}` immediately instead of blocking
  on the entire pipeline. Added `GET /analysis/{session_id}/status` for polling.
- `backend/backend/app/services/basket_analysis_service.py` — added
  `run_analysis_job()` (background-task entry point with its own DB session) and
  in-memory job status tracking. Also removed the old 14-second mid-loop cutoff
  that was silently truncating results (dropping fruit from the basket) to dodge
  an HTTP timeout that no longer applies now that this runs in the background.
- `frontend/index.html` — redirects to `scan.html` as soon as the session_id
  comes back (now near-instant), instead of waiting for full analysis.
- `frontend/scan.html` — polls `/analysis/{id}/status` every 1.5s (up to 2 min)
  and shows a real processing state instead of racing a single fetch against a
  hardcoded 5s timeout that failed outright on anything slower.

## 4. Batched grading
`backend/backend/app/services/grading_service.py` — added `grade_batch()`, which
groups crops by fruit_type and runs one `model.predict()` call per group instead
of one call per individual fruit (each `predict()` call has real overhead).
`backend/backend/app/services/basket_analysis_service.py` — the live pipeline
(the one actually wired to the frontend) now calls `grade_batch()` once instead
of `grade_fruit()` per fruit in a loop.

## 5. CUDA visibility
`backend/backend/app/main.py` — logs `CUDA available: True/False` at startup.
If it logs False, inference is CPU-bound and is likely your real floor on
processing time — worth knowing before spending more time on the above.

## Not touched
`backend/backend/app/api/process.py` implements a second, separate pipeline
(`/scan/process/{scan_id}`) that duplicates most of `basket_analysis_service.py`,
but nothing in the frontend calls it. Left alone — wasn't in scope, but worth
deciding whether it's dead code.

## Note on what's excluded from this zip
The `backend/backend/fruit_split/` training image dataset (~1400 JPEGs, ~400MB)
and `.git` history were stripped to keep this download small. Nothing in the
code changes touches or depends on them.
