"""Execute setup + code for every topic record (manual and supplemental).

Run after build_content.py. Writes outputs/cv_knowledge_base/reports/topic_validation.json.
"""
from pathlib import Path
import json, os, sys, tempfile, traceback
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'cv_knowledge_base'
sys.path.insert(0, str(ROOT / 'solutions'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_supplement import synthetic_image  # noqa: E402


def main():
    image = synthetic_image()
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    topics = [json.loads(l) for l in (ROOT / 'records' / 'topics.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    results, failed, skipped = [], 0, 0
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        for t in topics:
            if not t['code'].strip():
                skipped += 1
                results.append(dict(topic=t['id'], status='no_code'))
                continue
            ns = dict(cv=cv, np=np, image=image.copy(), g=g.copy(), print=lambda *a, **k: None)
            try:
                exec(t.get('setup', ''), ns)
                exec(t['code'], ns)
                results.append(dict(topic=t['id'], status='passed'))
            except Exception:
                failed += 1
                results.append(dict(topic=t['id'], status='failed', error=traceback.format_exc(limit=2)))
        os.chdir(cwd)
    report = dict(status='passed' if failed == 0 else 'failed', topics=len(topics), executed=len(topics) - skipped,
                  failed=failed, no_code=[r['topic'] for r in results if r['status'] == 'no_code'],
                  opencv=cv.__version__,
                  data='Synthetic image; confirms each fragment runs with its stated setup, not real-data accuracy.',
                  results=results)
    (ROOT / 'reports' / 'topic_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'topics', 'executed', 'failed', 'no_code')}))
    for r in results:
        if r['status'] == 'failed':
            print('FAIL', r['topic'], r['error'].strip().splitlines()[-1])


if __name__ == '__main__':
    main()
