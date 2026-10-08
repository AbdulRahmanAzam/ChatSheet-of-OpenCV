"""Cache per-model reader features for every eval set, then grid-search combination weights.

Usage:
    python work/tune_reader.py cache nli-deberta-v3-base     # slow: runs the reader once per question
    python work/tune_reader.py tune                          # fast: compares all cached models
Dev sets tune the weights. TEST sets (Lab 05, option-shuffled copies, frozen conceptual hold-out) only report scores with the dev-chosen weights.
"""
from pathlib import Path
import itertools, json, sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mcq_engine as E  # noqa: E402

CACHE = HERE / 'eval' / 'cache'
DEV = ['mcq_features_edges.txt', 'mcq_lab01_lab03.txt', 'mcq_lab06.txt', 'mcq_other_labs_selfwritten.txt']
TEST = ['mcq_lab05.txt', 'mcq_lab05.shuf.txt', 'mcq_lab06.shuf.txt', 'mcq_lab01_lab03.shuf.txt', 'holdout_conceptual.shuf.txt']


def rows(name):
    return [l.split('|') for l in (HERE / 'eval' / name).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]


def cache(model):
    engine = E.Engine(model_dir=HERE.parent / 'models' / model)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in DEV + TEST:
        if (CACHE / f'{model}__{name}.json').exists():
            continue
        feats = [engine.features(q, opts) for _, q, *opts, _ in rows(name)]
        (CACHE / f'{model}__{name}.json').write_text(json.dumps(feats), encoding='utf-8')
        print(model, name, 'cached', flush=True)


def score(model, name, w):
    engine = E.Engine.__new__(E.Engine)
    engine.w_lex, engine.w_nli, engine.w_ord = w
    feats = json.loads((CACHE / f'{model}__{name}.json').read_text(encoding='utf-8'))
    return sum('ABCD'[engine.combine(q, opts, f)['choice']] == key.strip() for (_, q, *opts, key), f in zip(rows(name), feats))


def tune():
    models = sorted({p.name.split('__')[0] for p in CACHE.glob('*.json')})
    grid = list(itertools.product([0.0, 0.15, 0.35, 0.6, 1.0], [1.0], [0.0, 0.3, 0.6, 1.0, 1.5]))
    for model in models:
        if not all((CACHE / f'{model}__{n}.json').exists() for n in DEV + TEST):
            continue
        best = max(grid, key=lambda w: sum(score(model, n, w) for n in DEV))
        dev = {n: score(model, n, best) for n in DEV}
        test = {n: score(model, n, best) for n in TEST}
        print(f"{model}: weights lex/nli/ord={best} dev {sum(dev.values())}/349 {dev} | test {test}")


if __name__ == '__main__':
    cache(sys.argv[2]) if sys.argv[1] == 'cache' else tune()
