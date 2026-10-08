"""Compare reader models on question sets: accuracy under a few fixed weightings, and speed.

Usage: python work/compare_readers.py eval/holdout_conceptual.shuf.txt [more files] --models=xsmall,small,base,large-wanli
Features are cached in work/eval/cache/, so re-running only re-combines.
"""
from pathlib import Path
import json, sys, time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mcq_engine as E  # noqa: E402

CACHE = HERE / 'eval' / 'cache'
WEIGHTS = {'reader only': (0.0, 1.0, 0.0), 'reader + words': (0.35, 1.0, 0.6), 'reader + light words': (0.15, 1.0, 0.3)}


def rows(path):
    return [l.split('|') for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]


def main(argv):
    files = [HERE / a if not Path(a).is_absolute() else Path(a) for a in argv if not a.startswith('--')]
    models = next(a.split('=', 1)[1] for a in argv if a.startswith('--models=')).split(',')
    CACHE.mkdir(parents=True, exist_ok=True)
    for m in models:
        name = f'nli-deberta-v3-{m}'
        engine, secs, n = None, 0.0, 0
        results = {w: 0 for w in WEIGHTS}
        total = 0
        for f in files:
            cpath = CACHE / f'{name}__{f.name}.json'
            data = rows(f)
            if cpath.exists():
                feats = json.loads(cpath.read_text(encoding='utf-8'))
            else:
                engine = engine or E.Engine(model_dir=HERE.parent / 'models' / name)
                t = time.time()
                feats = [engine.features(q, opts) for _, q, *opts, _ in data]
                secs += time.time() - t
                n += len(data)
                cpath.write_text(json.dumps(feats), encoding='utf-8')
            comb = E.Engine.__new__(E.Engine)
            for w, (a, b, c) in WEIGHTS.items():
                results[w] += sum('ABCD'[comb.combine(q, opts, ft, a, b, c)['choice']] == key.strip()
                                  for (_, q, *opts, key), ft in zip(data, feats))
            total += len(data)
        speed = f'{secs / n:.2f}s/question' if n else 'cached'
        print(f'{m:12s} ' + '  '.join(f'{w}: {v}/{total}' for w, v in results.items()) + f'  [{speed}]', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
