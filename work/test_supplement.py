"""Execute every supplemental topic snippet and check key outputs.

Run with OpenCV 4.13 on the path, e.g. PYTHONPATH=work/cv4;work/deps.
Writes outputs/cv_knowledge_base/reports/supplement_validation.json.
"""
from pathlib import Path
import json, os, tempfile, traceback
import cv2 as cv
import numpy as np
import supplement_topics as S

ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'cv_knowledge_base'


def synthetic_image():
    img = np.full((240, 320, 3), 40, np.uint8)
    cv.rectangle(img, (30, 30), (130, 110), (200, 180, 160), -1)
    cv.circle(img, (220, 80), 40, (60, 60, 230), -1)
    cv.rectangle(img, (170, 150), (300, 220), (90, 200, 90), -1)
    cv.putText(img, 'CV', (40, 200), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    for i in range(0, 320, 16):
        cv.line(img, (i, 120), (i + 8, 140), (150, 150, 150), 1)
    return img


# Extra assertions per topic, evaluated in the snippet's namespace.
CHECKS = {
    'fourier': "low_u8.shape == g.shape and np.abs(high).mean() > 0",
    'pyramids': "len(levels) == 4 and np.array_equal(rebuilt, g)",
    'template': "max_val > 0.99 and top_left == (20, 100)",
    'corners': "pts is not None and len(pts) > 0",
    'feature_detectors': "len(kp_fast) > 0 and des_a is not None",
    'flann': "len(good) > 5",
    'shape': "len(contours) > 0 and hu.shape == (7,)",
    'channels': "brighter.mean() >= g.mean() and padded.shape[0] == image.shape[0] + 20",
    'sharpening': "sharp.shape == image.shape",
    'noise': "cv.PSNR(g, nlm) > cv.PSNR(g, noisy)",
    'grabcut': "fg.max() == 255",
    'inpaint': "np.abs(restored.astype(int) - image.astype(int))[defect > 0].mean() < np.abs(damaged.astype(int) - image.astype(int))[defect > 0].mean()",
    'cascades': "not face.empty()",
    'background': "fgmask[40:60, 40:60].mean() > 200 and motion.max() == 255",
    'optical_flow': "len(good_new) > 0 and abs(float(np.median((good_new - good_old)[:, 0])) - 3) < 0.5 and flow.shape == g.shape + (2,)",
    'meanshift': "back.shape == g.shape and -1 <= similarity <= 1",
    'color_spaces': "dist.shape == g.shape and enhanced.shape == image.shape",
    'skeleton': "0 < cv.countNonZero(skel) < cv.countNonZero(bw)",
    'stereo': "valid.any()",
    'dnn': "blob.dtype == np.float32",
    'plotting': "os.path.exists('stages.png')",
}


def main():
    image = synthetic_image()
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    results, failed = [], 0
    ids = [t[0] for t in S.TOPICS]
    assert len(ids) == len(set(ids)), 'duplicate topic ids'
    assert set(CHECKS) == set(ids), set(ids) ^ set(CHECKS)
    for name in S.API_ROWS.strip().splitlines():
        assert hasattr(cv, name.split('|')[0]), name
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        for tid, *_rest in S.TOPICS:
            code = _rest[4]
            ns = dict(cv=cv, np=np, image=image.copy(), g=g.copy(), os=os, print=lambda *a, **k: None)
            try:
                exec(code, ns)
                ok = bool(eval(CHECKS[tid], ns))
                results.append(dict(topic=tid, executed=True, check=CHECKS[tid], passed=ok))
                failed += not ok
            except Exception:
                failed += 1
                results.append(dict(topic=tid, executed=False, error=traceback.format_exc(limit=2)))
        os.chdir(cwd)
    report = dict(status='passed' if failed == 0 else 'failed', topics=len(results), failed=failed,
                  opencv=cv.__version__, numpy=np.__version__,
                  data='Synthetic image only; checks confirm snippets run and produce sane output, not real-data accuracy.',
                  results=results)
    (ROOT / 'reports' / 'supplement_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'topics', 'failed', 'opencv')}))
    for r in results:
        if not r.get('passed'):
            print('FAIL', r['topic'], r.get('error', r.get('check')))


if __name__ == '__main__':
    main()
