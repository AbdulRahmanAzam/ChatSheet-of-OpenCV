"""Custom code for a computer-vision question: plan -> write -> check -> repair. Offline.

    python work/codegen.py "Write solve(img) that counts the coins and returns the largest radius"
    python work/codegen.py question.txt --out=answer.py

1. Plan: planner.py picks the method; its tested step code and the KB's API facts become reference context.
2. Write: a small local code model (Qwen2.5-Coder, 4-bit GGUF via llama.cpp) writes code for this exact question.
3. Check: the code must parse, import only cv2/numpy/matplotlib/math, use only cv2/numpy names that exist in the
   installed libraries, define the requested function, and run without error on a generic synthetic input.
4. Repair: on failure the exact error goes back to the model (up to 3 rounds). If it still fails, the tested
   planner script is returned instead, clearly marked.
"""
from pathlib import Path
import ast, json, re, subprocess, sys, tempfile, textwrap, time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planner  # noqa: E402

MODELS = HERE.parent / 'models' / 'qwen2.5-coder'
DEFAULT_MODEL = next((m for m in ('qwen2.5-coder-1.5b-instruct-q4_k_m.gguf', 'qwen2.5-coder-0.5b-instruct-q4_k_m.gguf')
                      if (MODELS / m).exists()), 'qwen2.5-coder-1.5b-instruct-q4_k_m.gguf')   # 1.5B: 11/20, 0.5B: 3/20
ALLOWED_IMPORTS = {'cv2', 'numpy', 'matplotlib', 'matplotlib.pyplot', 'math'}
SYSTEM = ('You are an expert OpenCV programmer. Write correct, simple Python using only OpenCV 4, numpy and math. '
          'Begin the code with exactly:\nimport cv2 as cv\nimport numpy as np\n'
          'Images are BGR numpy arrays. Reply with one ```python code block containing the imports and the requested '
          'function. Do not read files, do not show windows, do not add example usage.')

# Standard imports that are added automatically when the code uses the name but forgot the import.
AUTO_IMPORTS = {'cv': 'import cv2 as cv', 'cv2': 'import cv2', 'np': 'import numpy as np', 'math': 'import math'}


def auto_fix(code):
    """Mechanical fixes a compiler-style tool would make: add missing standard imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    used = {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    bound = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {a.asname or a.name.split('.')[0] for a in n.names}
    missing = [AUTO_IMPORTS[name] for name in AUTO_IMPORTS if name in used and name not in bound]
    return '\n'.join(missing + [code]) if missing else code

_LLM = {}


def llm(model=DEFAULT_MODEL):
    if model not in _LLM:
        from llama_cpp import Llama
        _LLM[model] = Llama(model_path=str(MODELS / model), n_ctx=4096, n_threads=None, verbose=False, seed=0)
    return _LLM[model]


# ---------------------------------------------------------------- 1. context from the plan and the KB
def context(question, max_steps=8):
    """Tested step code for the planned method + exact API facts for the OpenCV functions involved."""
    try:
        p = planner.build(question)
    except Exception:
        return '', None
    code = []
    for sid in p['steps'][:max_steps]:
        s = planner.STEPS[sid]
        # Keep only the OpenCV logic: the planner's display/report bookkeeping would be copied blindly.
        body = '\n'.join(l for l in s['code'].splitlines() if not l.lstrip().startswith(('stages[', 'report[', 'state[')))
        for name, (value, _) in p['params'].items():           # inline settings: no undefined constants to copy
            body = re.sub(r'\b' + name + r'\b', planner.fmt_value(value), body)
        if body.strip():
            code.append(f"# {s['title']}\n{body}")
    params = ''
    funcs = set(re.findall(r'cv\.([A-Za-z]\w+)\(', '\n'.join(code)))
    funcs |= set(re.findall(r'cv2?\.([A-Za-z]\w+)', question))
    api = [f['statement'] for f in planner.FACTS
           if f['source_refs'][0].startswith('opencv_docs') and any(re.search(r'\b' + fn + r'\(', f['statement']) for fn in funcs)]
    ctx = (f"Method chosen by the course planner: {p['title']}.\n"
           f"Tested reference code from the course (img is a BGR image, cv is cv2). Adapt it to the task and "
           f"leave out parts the task does not need:\n```python\n"
           + '\n\n'.join(code) + '\n```\n')
    if api:
        ctx += 'OpenCV API notes:\n' + '\n'.join('- ' + a for a in api[:8]) + '\n'
    return ctx, p


def prompt(question, ctx):
    return (ctx + '\n' if ctx else '') + f'Task:\n{question.strip()}\n'


def extract_code(text):
    m = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S)
    return (max(m, key=len) if m else text).strip()


# ---------------------------------------------------------------- 3. checks
def wanted_function(question):
    m = re.search(r'\b([a-z_]\w*)\(([^)]*)\)', question)
    if not m:
        return None, []
    return m.group(1), [a.strip() for a in m.group(2).split(',') if a.strip()]


def static_check(code, fname, args):
    """Problems found without running the code (empty list = fine)."""
    import cv2, numpy
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f'SyntaxError: {e.msg} at line {e.lineno}']
    problems, aliases = [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    problems.append(f'import {a.name} is not allowed; use only cv2, numpy, math')
                aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom):
            if (node.module or '') not in ALLOWED_IMPORTS:
                problems.append(f'from {node.module} import ... is not allowed; use only cv2, numpy, math')
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            mod = aliases.get(node.value.id)
            lib = {'cv2': cv2, 'numpy': numpy}.get(mod)
            if lib is not None and not hasattr(lib, node.attr):
                problems.append(f'{node.value.id}.{node.attr} does not exist in {mod} {getattr(lib, "__version__", "")}')
    if fname:
        defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        if fname not in defs:
            problems.append(f'the function {fname}({", ".join(args)}) is not defined')
        elif len(defs[fname].args.args) != len(args):
            problems.append(f'{fname} must take exactly {len(args)} argument(s): {", ".join(args)}')
    return problems


# Generic inputs by argument name: they test that the code runs, they are NOT the graded test inputs.
PROBE = textwrap.dedent('''
    import sys, json, numpy as np, cv2
    rng = np.random.default_rng(0)
    def picture(h=240, w=320):
        img = np.full((h, w, 3), 90, np.uint8)
        cv2.circle(img, (80, 80), 30, (220, 220, 220), -1)
        cv2.rectangle(img, (150, 60), (260, 180), (40, 200, 230), -1)
        cv2.line(img, (20, 220), (300, 220), (255, 255, 255), 3)
        return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    def value(name):
        n = name.lower()
        if n in ('signal', 'data', 'values', 'readings', 'x'):
            return np.sin(np.linspace(0, 20, 500)) + rng.normal(0, .1, 500)
        if 'corner' in n or 'point' in n or 'pts' in n:
            return [[40, 30], [280, 40], [290, 200], [30, 210]]
        if n in ('images', 'imgs', 'frames'):
            return [picture(), picture()]
        if n in ('k',):
            return 3
        return picture()
    ns = {}
    exec(compile(open(sys.argv[1], encoding='utf-8').read(), 'answer.py', 'exec'), ns)
    fn, args = sys.argv[2], json.loads(sys.argv[3])
    result = ns[fn](*[value(a) for a in args])
    if result is None:
        raise ValueError(fn + ' returned None; it must return the requested result')
    print('RAN OK', type(result).__name__)
''')


def run_check(code, fname, args, timeout=60):
    """Run the function once on generic synthetic input in a separate process; return error text or ''."""
    if not fname:
        return ''
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'answer.py').write_text(code, encoding='utf-8')
        Path(d, 'probe.py').write_text(PROBE, encoding='utf-8')
        try:
            r = subprocess.run([sys.executable, 'probe.py', 'answer.py', fname, json.dumps(args)], cwd=d,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f'the function did not finish within {timeout} s (infinite loop?)'
    if r.returncode == 0:
        return ''
    lines = [l for l in r.stderr.strip().splitlines() if l.strip() and 'Warning' not in l]
    where = [l.strip() for l in lines if 'answer.py' in l]
    return (where[-1] + '\n' if where else '') + (lines[-1] if lines else 'unknown error')


# ---------------------------------------------------------------- 2 + 4. write and repair
def generate(question, model=DEFAULT_MODEL, use_context=True, repair=3, max_tokens=900):
    """Returns dict(code, ok, attempts, log, source, seconds)."""
    t0 = time.time()
    fname, args = wanted_function(question)
    ctx, plan = context(question) if use_context else ('', None)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt(question, ctx)}]
    log, code = [], ''
    for attempt in range(1 + repair):
        out = llm(model).create_chat_completion(messages=messages, temperature=0.0 if attempt == 0 else 0.3,
                                                max_tokens=max_tokens, seed=attempt)
        reply = out['choices'][0]['message']['content']
        code = auto_fix(extract_code(reply))
        problems = static_check(code, fname, args)
        error = '\n'.join(problems) if problems else run_check(code, fname, args)
        log.append(error or 'passed')
        if not error:
            return dict(code=code, ok=True, attempts=attempt + 1, log=log, source='generated',
                        seconds=round(time.time() - t0, 1))
        if attempt < repair:
            messages += [{'role': 'assistant', 'content': f'```python\n{code}\n```'},
                         {'role': 'user', 'content': f'That code fails:\n{error}\nFix it. Reply with the full corrected code.'}]
    fallback = planner.to_script(plan) if plan else code
    return dict(code=fallback if plan else code, ok=False, attempts=1 + repair, log=log,
                source='planner fallback' if plan else 'generated (unverified)', seconds=round(time.time() - t0, 1),
                last_generated=code)


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], True) for a in argv if a.startswith('--'))
    q = Path(args[0]).read_text(encoding='utf-8') if args and Path(args[0]).is_file() else ' '.join(args)
    r = generate(q, model=opts.get('model', DEFAULT_MODEL))
    header = (f"# {'Checked' if r['ok'] else 'NOT verified, showing the tested planner script instead'}: "
              f"{r['source']}, {r['attempts']} attempt(s), {r['seconds']} s\n")
    text = header + r['code'] + '\n'
    if 'out' in opts:
        Path(opts['out']).write_text(text, encoding='utf-8'); print('code ->', opts['out'])
    else:
        print(text)


if __name__ == '__main__':
    main(sys.argv[1:])
