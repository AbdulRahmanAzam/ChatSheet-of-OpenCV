"""Run the MCQ answerer over every key file in work/eval and write reports/mcq_eval.json."""
from pathlib import Path
import json, sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mcq_answer import Index, answer, load_passages, KB  # noqa: E402

NOTES = {
    'mcq_lab01_lab03.txt': 'User-supplied Lab 01/03 MCQs. Held-out: facts written before seeing them (first run 75/100); scorer fixes (x-sign, NOT/all-of handling, stemming) applied after.',
    'mcq_features_edges.txt': 'User-supplied Lab 04 MCQs with answer key. Facts were written after seeing these, so the score is optimistic.',
    'mcq_other_labs_selfwritten.txt': 'Written by the KB author for Labs 01/03/05/06; a pipeline sanity check, not an independent benchmark.',
}


def main():
    index = Index(load_passages())
    report = {}
    for path in sorted((HERE / 'eval').glob('mcq_*.txt')):
        rows = [l.split('|') for l in path.read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]
        wrong = []
        for n, (section, q, *opts, key) in enumerate(rows, 1):
            k, _, evidence = answer(index, q, opts)
            if 'ABCD'[k] != key.strip():
                wrong.append(dict(n=n, section=section, question=q, picked=opts[k], key=opts['ABCD'.index(key.strip())], evidence=evidence))
        report[path.name] = dict(questions=len(rows), correct=len(rows) - len(wrong), note=NOTES.get(path.name, ''), wrong=wrong)
        print(f"{path.name}: {len(rows) - len(wrong)}/{len(rows)}")
    (KB / 'reports' / 'mcq_eval.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
