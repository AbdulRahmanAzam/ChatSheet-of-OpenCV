"""Does the planner choose the right method? Checks recipe (and a required step) per task text.

    python work/plan_eval.py eval/plan_dev.txt eval/plan_holdout.txt
Each generated script is also compiled, so a plan with broken code counts as wrong.
"""
from pathlib import Path
import sys, types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planner  # noqa: E402
import plan_selftest as T  # noqa: E402


def smoke(p):
    """Generate the script and run it on a generic synthetic input of the right kind (must not crash)."""
    mod = types.ModuleType('generated')
    exec(compile(planner.to_script(p), '<plan>', 'exec'), mod.__dict__)
    img = T.texture(240, 320, 11)
    kind = p['kind'].replace('video+', '')
    if kind == 'image':
        out = mod.run(img)
    elif kind == 'pair':
        out = mod.run(img[40:200, 60:260].copy(), img)
    elif kind == 'images':
        out = mod.run([img[:, :200].copy(), img[:, 120:].copy()])
    elif kind == 'video':
        state = {}
        for f in T.zone_frames()[:6]:
            out = mod.run(f, state)
    else:
        out = mod.run(T.sensor()[0])
    assert isinstance(out[1], dict)


def evaluate(path):
    rows = [l.split('|', 1) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]
    ok, wrong = 0, []
    for expect, text in rows:
        recipe, _, step = expect.partition(':')
        try:
            p = planner.build(text)
            smoke(p)
            good = p['recipe'] == recipe and (not step or step in p['steps'])
            got = p['recipe'] + ('' if not step else f" steps={p['steps']}")
        except Exception as e:
            good, got = False, f'ERROR {type(e).__name__}: {e}'
        ok += good
        if not good:
            wrong.append(f'  expected {expect:28s} got {got}\n      {text[:110]}')
    print(f'{Path(path).name}: {ok}/{len(rows)} right method')
    print('\n'.join(wrong))
    return ok, len(rows)


if __name__ == '__main__':
    files = sys.argv[1:] or ['eval/plan_dev.txt', 'eval/plan_holdout.txt', 'eval/plan_holdout2.txt']
    results = [evaluate(HERE / f if not Path(f).is_absolute() else f) for f in files]
    # The build gate: every course lab task must get the right method (hold-outs are reported, not gated).
    sys.exit(0 if results[0][0] == results[0][1] else 1)
