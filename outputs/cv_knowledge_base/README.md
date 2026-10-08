# Offline computer-vision knowledge base — version 0.1

Open **[index.html](index.html)** in your browser. Its search works from local files without an internet connection, account, API key, model download, or running server.

This deliverable completes the knowledge-base stage for the supplied material. It contains the original sources, searchable extraction, curated explanations, corrections, and worked exercise examples. **No planning engine, code-completion model, training pipeline, or installable prediction library has been implemented.** Those stages await your instruction.

## What is included

- All 10 supplied PDFs, preserved byte-for-byte, with SHA-256 hashes.
- All 135 pages rendered to images, with local OCR and word-position records.
- 82 topic records: 61 from the supplied manuals plus 21 tested supplemental related topics (Fourier, pyramids, template matching, corners, FAST/AKAZE/BRISK, FLANN, shape descriptors, channels/LUT/borders, sharpening, noise/PSNR, GrabCut, inpainting, Haar cascades, background subtraction, optical flow, mean-shift/CamShift, LAB/YCrCb, skeletons, stereo depth, cv.dnn, Matplotlib plotting).
- 41 worked exercise solutions, including the material-texture task embedded in Lab 04.
- 38 manual code example groups, mapped to source scans and corrected runnable equivalents.
- 164 question-and-answer entries derived from the curated topics, plus 143 API signature entries checked against installed OpenCV 4.13, NumPy and Matplotlib.
- A SQLite full-text index, JSON/JSONL records, a readable handbook, a corrections log, and synthetic demonstrations.

## Start reading

| File | Purpose |
|---|---|
| [index.html](index.html) | Offline reference browser and search |
| [HANDBOOK.md](HANDBOOK.md) | Explanations, uses, pitfalls and short code examples |
| [TASK_SOLUTIONS.md](TASK_SOLUTIONS.md) | Source-linked steps, assumptions and solution entry points for every task |
| [MANUAL_CODE_INVENTORY.md](MANUAL_CODE_INVENTORY.md) | Locate original code scans and corrected equivalents |
| [reports/CORRECTIONS.md](reports/CORRECTIONS.md) | Source errors and corrected interpretations |
| [reports/COVERAGE.md](reports/COVERAGE.md) | Coverage, missing inputs, size and verification results |
| [reports/validation.json](reports/validation.json) | Actual example test results and measured synthetic outcomes |
| [reports/supplement_validation.json](reports/supplement_validation.json) | Execution results for every supplemental topic snippet |
| [reports/topic_validation.json](reports/topic_validation.json) | Execution results for all topic fragments with their setup |
| [reports/search_eval.json](reports/search_eval.json) | Retrieval accuracy on tuned and held-out questions |
| [reports/mcq_eval.json](reports/mcq_eval.json) | Multiple-choice answering results and misses |
| [records/SCHEMA.md](records/SCHEMA.md) | Data organization for a later planner/predictor |

## Optional Python lookup

Python 3.10+ with SQLite FTS5 is sufficient; no third-party library is needed for lookup.

```text
python search.py "watershed markers" --limit 5
python search.py "BFMatcher" --json
python search.py "student grades" --appendix
python search.py "Canny" --raw
```

Default lookup uses curated core-domain records. `--raw` includes unverified OCR; `--appendix` includes the unrelated introductory Python tasks and the one-dimensional sensor exercise. Lookup returns records; it does not generate novel answers or predict tokens.

## Worked examples

The Python files under `solutions/` are reference answers requested for the knowledge base. The demonstrations in `demos/` are already generated and viewable offline.

To rerun them, install Python and the packages in `requirements-examples.txt` once, then run:

```text
python solutions/run_synthetic.py
```

For an offline machine, prepare matching package wheels on a connected machine first:

```text
python -m pip download -r requirements-examples.txt -d wheels
python -m pip install --no-index --find-links wheels -r requirements-examples.txt
```

The preparation machine must target the offline machine's operating system, architecture, and Python version. The knowledge-base archive does **not** bundle Python or package wheels. Once installed, the example code uses local files and has no network calls. Do not install several OpenCV wheel variants in the same environment; this set uses headless OpenCV and saves plots instead of calling a desktop image window.

The tested version is OpenCV 4.13.0. A preliminary test with the available OpenCV 5.0 wheel found `HOGDescriptor` unavailable, so the reference environment is pinned rather than claiming compatibility with all versions. The optional Pandas task variant is retained but Pandas is not a core dependency and that variant was not tested.

## How source quality is handled

Original PDFs and page scans are the authoritative record of what the manuals contain. OCR preserves all pages but has recognition errors, especially in code, equations and figures. It is **not a fully proofread verbatim transcription**. Manual examples in `solutions/manual_examples.py` are corrected reconstructions, not exact transcriptions; substitutions for scikit-image HOG/LBP are explicitly marked.

Do not train a later model directly on unreviewed OCR. The curated layer separates source claims, corrections, assumptions, and additional topics. Course submission/upload instructions are retained as document content; they were not executed. No source material was uploaded or published.

## What is still needed from you

- Lab 02 Manual, if it exists.
- A separate Lab 04 task sheet, if it exists; the embedded task is already included.
- Original exercise images, videos, reference objects, point correspondences and any labeled datasets needed for actual-data evaluation.
- Clarification of “R10” and “botch filter” if these mean something other than ROI and box filter. Those possible interpretations are recorded as uncertain.

The corpus covers the supplied material and relevant fundamentals. It cannot guarantee answers to every possible computer-vision question. No real-world accuracy or full future runtime/RAM budget is claimed from the knowledge-base size alone.

## Rights and external references

The supplied course PDFs retain their original ownership. Keeping them in this private working copy does not establish permission to redistribute them in a public pip package. External sources are linked with short authored notes in `records/external_sources.json`; their websites are not mirrored. Runnable solution code and explanations were authored for this task.

## Rebuilding

Build scripts live in `../../work/`. From the project root, with the project `.venv` (Python 3.12, OpenCV 4.13 via `work/cv4;work/deps`):

```text
.venv/Scripts/python.exe work/build_all.py
```

It runs, in order and stopping on failure: supplemental snippet tests, the 41-task synthetic validation, API/content/index builds, execution of every topic's `setup` + `code`, the search and MCQ evaluations, the planner self-test and method-choice evaluation, the plan library (`plans/`) and `reports/COVERAGE.md`.

To add knowledge: manual facts go in `work/facts_manuals.py` / `work/facts_labNN.py`; new topics go in `work/supplement_topics.py` (plus a check in `work/test_supplement.py`); prerequisite code and paraphrase aliases for any topic go in `work/topic_setups.py`. Then rerun `build_all.py`.

## Planner: task -> steps -> code

`work/planner.py` turns a written computer-vision task into a step-by-step plan, offline and without a language model:

```text
.venv/Scripts/python.exe work/planner.py "Detect computer screens in a lab with Hough lines and report missing ones"
.venv/Scripts/python.exe work/planner.py task.txt --out=plan.md --script=solution.py
```

Each plan states the goal, input and chosen method (and the cue words that chose it), the limits to mention, then every
step with why it is there, its tunable parameters, its code, a warning, the course-manual facts behind it (with page) and
the closest worked lab task with its tested code. The full script at the end is assembled from the same step code, so plan
and script always agree. `plans/PLANS.md` has a plan and script for every course lab task.

How it decides: `work/plan_recipes.py` holds 21 recipes (screens, lanes, lines, circles, SIFT recognition, panorama, counting,
watershed, k-means, HSV color, thresholding, Canny, region growing, zone monitoring, enhancement, fusion, transforms, HOG/LBP,
Harris, wavelet signals, basic operations) with weighted cue words; `work/plan_steps.py` holds the 64 steps they are built
from (each declares what it needs and makes, and the planner checks the chain). Numbers in the task (K=4, 25x25, 45 degrees,
yellow, 12 screens, 40 pixels right) become the script's parameters. If no cue fires, the knowledge-base topic search decides.

How it is checked:
- `work/plan_selftest.py` runs 20 generated scripts on synthetic images/videos/signals with known answers (e.g. 5 screens
  drawn with one missing: must report 5 found, 3 ON, 2 OFF, missing at row 2 column 3). 20/20 pass.
- `work/plan_eval.py`: right method for all 35 lab tasks, and for two frozen hold-out sets of 40 new task wordings each
  (40/40 and 40/40; the first hold-out was used to find general weaknesses, the second was written before those fixes and
  never tuned on). Every plan's script is also executed on a generic input and must not crash.
- Limits: recipes cover the course's methods; a task needing a method outside them gets the nearest recipe. Synthetic
  tests prove the code logic, not tuning for real photos: start from the PARAMETERS block on real images.

## Ready for the next stages

- The planner's steps (`work/plan_steps.py`) are self-contained, executed code units with named inputs/outputs: the training
  and lookup material for the code predictor.

- Every topic record has `setup` (earlier steps it assumes), `requires` (names defined) and `code`, so a planner can chain steps and a predictor can train on self-contained, executed examples.
- `records/facts.jsonl` holds 349 atomic facts restated from all five manuals, each with its page and, where the manual is imprecise, a `note` with the precise view. `work/mcq_answer.py` answers multiple-choice questions offline from them (`reports/mcq_eval.json`).
- `search.py` is the retrieval baseline the planner can call (`from search import search`). On unseen paraphrases it ranks the right topic first about half the time (`reports/search_eval.json`); a learned task classifier is the planned improvement.
- The offline browser `index.html` uses its own simpler in-page search without stemming.
