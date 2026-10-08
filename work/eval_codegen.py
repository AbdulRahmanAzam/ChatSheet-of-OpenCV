"""Grade the code generator on the frozen custom tasks (work/eval/code_tasks.py).

    python work/eval_codegen.py [--model=FILE.gguf] [--no-context] [--repair=3] [--only=id1,id2]
A task passes only if the generated function returns the right answer on the task's own synthetic input.
"""
from pathlib import Path
import json, subprocess, sys, tempfile, time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'eval'))
import codegen  # noqa: E402
import code_tasks as C  # noqa: E402

GRADER = '''
import sys
sys.path.insert(0, sys.argv[3])
import code_tasks as C
ns = {}
exec(compile(open(sys.argv[1], encoding='utf-8').read(), 'answer.py', 'exec'), ns)
task = next(t for t in C.TASKS if t['id'] == sys.argv[2])
fname = task['sig'].split('(')[0]
result = ns[fname](*task['make']())
print('PASS' if task['check'](result) else 'WRONG ANSWER: ' + repr(result)[:200])
'''


def grade(code, task, timeout=90):
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'answer.py').write_text(code, encoding='utf-8')
        Path(d, 'grader.py').write_text(GRADER, encoding='utf-8')
        try:
            r = subprocess.run([sys.executable, 'grader.py', 'answer.py', task['id'], str(HERE / 'eval')], cwd=d,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 'TIMEOUT'
    out = r.stdout.strip().splitlines()
    if r.returncode == 0 and out:
        return out[-1]
    err = [l for l in r.stderr.strip().splitlines() if l.strip() and 'Warning' not in l]
    return 'ERROR: ' + (err[-1] if err else 'unknown')


def main(argv):
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], True) for a in argv if a.startswith('--'))
    model = opts.get('model', codegen.DEFAULT_MODEL)
    use_ctx = not opts.get('no-context')
    repair = int(opts.get('repair', 3))
    only = set(opts['only'].split(',')) if 'only' in opts else None
    tag = f"{Path(model).stem}{'' if use_ctx else '-noctx'}-repair{repair}"
    results, t0 = [], time.time()
    for task in C.TASKS:
        if only and task['id'] not in only:
            continue
        question = task['prompt']
        r = codegen.generate(question, model=model, use_context=use_ctx, repair=repair)
        code = r['code'] if r['ok'] else r.get('last_generated', r['code'])   # grade the model's own code
        verdict = grade(code, task)
        results.append(dict(id=task['id'], verdict=verdict, checked=r['ok'], attempts=r['attempts'], seconds=r['seconds'], code=code))
        print(f"{task['id']:16s} {'PASS' if verdict == 'PASS' else 'fail'}  checks={'ok' if r['ok'] else 'failed'} "
              f"attempts={r['attempts']} {r['seconds']}s  {'' if verdict == 'PASS' else verdict[:110]}", flush=True)
    n = len(results); passed = sum(x['verdict'] == 'PASS' for x in results)
    print(f'\n{tag}: {passed}/{n} correct answers, {sum(x["checked"] for x in results)}/{n} passed the checks, '
          f'{(time.time() - t0) / max(n, 1):.0f} s per task')
    out = HERE / 'eval' / 'codegen_results'
    out.mkdir(exist_ok=True)
    (out / f'{tag}.json').write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1:])
