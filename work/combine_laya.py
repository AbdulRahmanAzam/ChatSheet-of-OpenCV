"""Combine the base engine with Laya's option probabilities; choose the Laya weight on the 428 dev questions only.

    python work/combine_laya.py C:/Users/azama/laya-env/laya_probs.json
"""
from pathlib import Path
import json, sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mcq_engine as E  # noqa: E402
import mcq_answer as lex  # noqa: E402

DEV = ['mcq_lab01_lab03.shuf.txt', 'mcq_features_edges.shuf.txt', 'mcq_lab05.shuf.txt', 'mcq_lab06.shuf.txt']
HOLD = ['holdout_conceptual.shuf.txt']
WEIGHTS = [0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10]


def rows(name):
    return [l.split('|') for l in (HERE / 'eval' / name).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]


def score(names, probs, w, use_triples=True):
    eng = E.Engine.__new__(E.Engine)
    eng.w_lex, eng.w_nli, eng.w_ord = 0.15, 1.0, 0.3
    ok = n = 0
    for name in names:
        feats = json.loads((HERE / 'eval' / 'cache' / f'nli-deberta-v3-base__{name}.json').read_text(encoding='utf-8'))
        for (_, q, *opts, key), f, p in zip(rows(name), feats, probs[name]):
            if use_triples and f['triple'] is not None:
                choice = f['triple']['choice']
            else:
                base = eng.combine(q, opts, dict(f, triple=None))
                total = np.array(base.get('scores', [0] * len(opts)), float)
                sign = -1 if lex.is_negated(q) else 1          # NOT questions pick the lowest total
                total = total + sign * w * np.array(p[:len(opts)])
                choice = lex.pick(q, opts, list(total)) if w < 1e6 else int(np.argmax(p))
            ok += 'ABCD'[choice] == key.strip(); n += 1
    return ok, n


def main(path):
    probs = json.loads(Path(path).read_text())
    print('w_laya   dev(428)   holdout(40)')
    dev = {w: score(DEV, probs, w)[0] for w in WEIGHTS}
    for w in WEIGHTS:
        print(f'{w:6}   {dev[w]:4d}       {score(HOLD, probs, w)[0]:2d}')
    best = max(WEIGHTS, key=lambda w: (dev[w], -w))
    print(f'chosen on dev: w_laya={best} -> dev {dev[best]}/428, holdout {score(HOLD, probs, best)[0]}/40')
    print(f'Laya alone (triples still first): dev {score(DEV, probs, 1e9)[0]}/428, holdout {score(HOLD, probs, 1e9)[0]}/40')


if __name__ == '__main__':
    main(sys.argv[1])
