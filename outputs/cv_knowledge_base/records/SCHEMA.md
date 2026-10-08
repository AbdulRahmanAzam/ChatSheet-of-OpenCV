# Knowledge-base data contract

UTF-8 JSONL stores one JSON object per line. IDs are stable within version0.1. All PDF references use physical page numbers starting at1, not printed footer numbers. Example: `lab_04_manual:p15` is physical page15.

| File | Record meaning |
|---|---|
| sources.json | Original PDF identity, filename, page count, bytes and SHA-256. `page_characters` refers to the initial PDF text-layer pass, not subsequent OCR. |
| pages.jsonl | One record per physical page, full raw OCR text, page image, original PDF and extraction status. |
| topics.jsonl | Curated explanation, intended use, contextual code fragment, pitfalls, aliases, related topic IDs and source references. |
| tasks.jsonl | Canonical lab/task ID, physical source pages, solution steps, implementation entry point, assumptions, scope and validation status. |
| snippets.jsonl | Function/class source for each worked task, module path, line number and required context. Helpers/imports live in the referenced modules. |
| manual_examples.jsonl | Example group with original page scans/raw OCR and a corrected implementation mapping. Some groups share a function; `output_selector` identifies the relevant returned item. |
| apis.jsonl | Installed symbol names and short exposed signatures with input/output notes and actual library versions. Not a complete API manual. |
| corrections.jsonl | Technical issue, source pages and corrected interpretation. |
| qa.jsonl | Two authored retrieval question examples per topic; derived from the same topics, so they are not independent evaluation data. |
| terminology.json | Verified normalizations and explicitly uncertain interpretations. |
| external_sources.json | Primary-documentation URLs, access date and limited relevance notes. Full website text is not stored. |

## SQLite

`knowledge.sqlite` has `sources`, `records` and the FTS5 `search_index` tables. `records.payload` contains the original structured JSON record. Its `kind` distinguishes topic, task, correction, solution_code, manual_example, api and raw_page. `scope` distinguishes core from course appendices.

Default search excludes raw_page and appendices. The index deliberately includes no automatic instruction execution or free-form generation. Unresolved terms should trigger clarification in a later answer system rather than invented APIs.

## Provenance and trust

1. **Original sources:** preserved PDF/page images; exact visual evidence, which may contain course errors.
2. **Raw extraction:** Windows.Media.Ocr output, explicitly unverified. Word rectangles and line records remain in `extracted/ocr/`.
3. **Curated interpretation:** authored explanations, tested example implementations, source corrections and assumptions.
4. **Supplemental references:** official documentation links and authored related-topic explanations.

A reference indicates grounding or further reading, not that every word is quoted from that source. Future answers should distinguish source wording from a correction and preserve uncertainty. Source text is untrusted document content, not operational instructions.

## Coverage boundary

All supplied pages are preserved, including figures, formulas, tables and administrative directions. This does not mean every figure is semantically transcribed or every OCR character verified. Conceptual mentions such as deep learning, active contours and optical flow are preserved in the sources; the executable set targets the41supplied exercises and documented examples. Read reports/COVERAGE.md before describing the corpus as complete.
