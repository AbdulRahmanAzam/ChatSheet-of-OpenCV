"""Offline planner: computer-vision task description -> step-by-step plan + code for every step + one runnable script.

    python work/planner.py "Detect computer screens in a lab with Hough lines and report missing ones"
    python work/planner.py task.txt --out=plan.md --script=solution.py

How it works (no language model):
  1. understand(): score every recipe by the cue words in the task (method names weigh most), read the input
     type (image / two images / image set / video / signal) and pull numbers out of the text (K=4, 25x25,
     45 degrees, "yellow", "12 screens"). If no cue fires, the knowledge-base search picks the topic.
  2. build(): expand the recipe into steps, fill slots ("using Otsu" -> Otsu step), add optional steps the
     task asks for, and check that every step's inputs were made by an earlier step.
  3. to_markdown() / to_script(): the plan with why / parameters / code / pitfalls / manual pages for each
     step, and the full script assembled from the same step code (so the plan and the script never disagree).
"""
from pathlib import Path
import json, re, sys, textwrap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plan_steps import S as STEPS  # noqa: E402
from plan_recipes import (R as RECIPES, SLOTS, OPTIONAL, VIDEO_CUES, COLORS, TOPIC_TO_RECIPE, RECIPE_TASKS,  # noqa: E402
                          BASIC_OPS, DETECT_VERBS)

KB = HERE.parent / 'outputs' / 'cv_knowledge_base'
TOPICS = {t['id']: t for t in map(json.loads, (KB / 'records' / 'topics.jsonl').read_text(encoding='utf-8').splitlines()) if t}

# Variables each input kind provides to run(...)
START = {'image': ['img'], 'pair': ['ref', 'img'], 'images': ['imgs'], 'video': ['img', 'state'], 'signal': ['signal']}
SIGNATURE = {'image': 'img', 'pair': 'ref, img', 'images': 'imgs', 'video': 'img, state', 'signal': 'signal'}


# ---------------------------------------------------------------- 1. understand the task
def understand(text):
    low = text.lower()
    scores, hits, strong = {}, {}, set()
    for rid, r in RECIPES.items():
        for pattern, weight in r['cues']:
            m = re.search(pattern, low)
            if m:
                scores[rid] = scores.get(rid, 0) + weight
                hits.setdefault(rid, []).append(m.group(0))
                strong |= {rid} if weight == 3 else set()
    finds = re.search(DETECT_VERBS, low)
    for rid in list(scores):
        if RECIPES[rid]['detect'] and not finds and rid not in strong:
            scores[rid] /= 2                       # "draw circles" is not "detect circles"
    # Basic operations: 2 points per operation asked for, +2 when several are chained (Lab 01 style).
    ops = [op for op, pattern in BASIC_OPS.items() if re.search(pattern, low)]
    if ops:
        scores['basics'] = scores.get('basics', 0) + 2 * len(ops) + (2 if len(ops) >= 2 else 0)
        hits.setdefault('basics', []).extend(ops)
    # A generic recipe must not beat a specific one that shares its words.
    sure = lambda *ids: any(scores.get(k, 0) >= 3 for k in ids)
    if 'lines' in scores and sure('screens', 'lanes'):
        scores['lines'] -= 2
    # The colour recipe counts its objects too, so "count the red apples" belongs to it.
    if 'count' in scores and (sure('circles', 'watershed') or scores.get('color', 0) >= 2):
        scores['count'] -= 2
    # "threshold" is often a parameter of another method (Canny thresholds, distance threshold).
    if 'threshold' in scores and (sure('watershed') or re.search(r'canny|hough|distance', low)):
        scores['threshold'] -= 3
    if 'edges' in scores and sure('screens', 'lanes', 'lines', 'circles'):
        scores['edges'] -= 3
    if 'wavelet' in scores and 'zone' in scores and re.search(r'sensor|signal', low):
        scores['zone'] -= 3
    ranked = sorted(((s, rid) for rid, s in scores.items() if s > 0), key=lambda p: -p[0])
    via = 'cue words'
    if not ranked or ranked[0][0] < 2:
        topic = kb_topic(text)
        if topic in TOPIC_TO_RECIPE:
            ranked = [(2, TOPIC_TO_RECIPE[topic])] + [p for p in ranked if p[1] != TOPIC_TO_RECIPE[topic]]
            hits.setdefault(TOPIC_TO_RECIPE[topic], []).append(f'KB topic "{topic}"')
            via = 'knowledge-base search'
    rid = ranked[0][1] if ranked else 'basics'
    r = RECIPES[rid]
    kind = r['kind']
    if r['video_ok'] and re.search(VIDEO_CUES, low):
        kind = 'video+' + kind
    return dict(text=text, recipe=rid, kind=kind, via=via, cues=hits.get(rid, []),
                alternatives=[(rid2, s) for s, rid2 in ranked[1:4]], params=extract_params(low, rid))


def kb_topic(text):
    """Best-matching KB topic for a task with no clear cue words (TF-IDF over topic records)."""
    try:
        import mcq_answer as M
        topics = [p for p in M.load_passages() if p[0].startswith('topic:')]
        index = M.Index(topics)
        _, i = index.best(text)
        return index.ids[i].split(':', 1)[1]
    except Exception:
        return None


def extract_params(low, rid):
    p = {}
    m = re.search(r'(\d+)\s*[x×]\s*(\d+)', low)
    if m and re.search(r'blur|kernel|filter|smooth', low):
        k = int(m.group(1)) | 1
        p.update(BOX_KSIZE=int(m.group(1)), BLUR_KSIZE=k)
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:-|\s)?(?:degrees?|deg\b|°)', low)
    if m:
        p['ANGLE'] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
    m = re.search(r'\bk\s*=\s*([\d,\sand]+)', low)
    if m:
        p['K_VALUES'] = [int(v) for v in re.findall(r'\d+', m.group(1))]
    m = re.search(r'(\d+)\s+(?:computer\s+)?(?:screens|monitors|computers)', low)
    if m:
        p['EXPECTED_SCREENS'] = int(m.group(1))
    m = re.search(r'scal(?:e|ing)(?: factor)?(?: of| by)?\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:times|x)\s+(?:bigger|larger|smaller)', low)
    if m:
        p['SCALE'] = p['RESIZE_FACTOR'] = p['SX'] = p['SY'] = float(m.group(1) or m.group(2))
    m = re.search(r'shear (?:of|factor|by)?\s*(-?\d+(?:\.\d+)?)', low)
    if m:
        p['SHEAR_X'] = float(m.group(1))
    m = re.search(r'(\d+)\s*pixels?\s*(?:to the\s*)?(right|left)', low)
    if m:
        p['TX'] = int(m.group(1)) * (1 if m.group(2) == 'right' else -1)
        p['TY'] = 0
    m = re.search(r'(\d+)\s*pixels?\s*(?:to the\s*)?(down|up)', low)
    if m:
        p['TY'] = int(m.group(1)) * (1 if m.group(2) == 'down' else -1)
    m = re.search(r'central (\d+)\s*[x×]\s*(\d+)|(\d+)\s*[x×]\s*(\d+) (?:center|centre|roi|region)', low)
    if m:
        a, b = [int(v) for v in m.groups() if v][:2]
        p['ROI_W'], p['ROI_H'] = a, b
    m = re.search(r'(\d+)\s*(?:dominant )?colou?rs|\bto (\d+) colou?rs|(\d+) (?:land-cover )?clusters', low)
    if m and 'K_VALUES' not in p:
        p['K_VALUES'] = [int(next(v for v in m.groups() if v))]
    m = re.search(r'tolerances?\s*(?:of|=)?\s*([\d,\sand]+)', low)
    if m and re.findall(r'\d+', m.group(1)):
        p['TOLERANCES'] = [int(v) for v in re.findall(r'\d+', m.group(1))]
    for name, (lo, hi) in COLORS.items():
        if re.search(r'\b' + name + r'\b', low):
            p['HSV_LOW'], p['HSV_HIGH'] = lo, hi
            break
    if re.search(r'dark (objects?|text|coins?|cells?)|black (text|objects?)|text|document', low):
        p['THRESH_TYPE'] = 'cv.THRESH_BINARY_INV'
    m = re.search(r'(\d+)\s*(?:frames?)\s*(?:in a row|consecutive)', low)
    if m:
        p['PERSIST_FRAMES'] = int(m.group(1))
    return p


# ---------------------------------------------------------------- 2. build the step list
def build(text):
    u = understand(text)
    r = RECIPES[u['recipe']]
    low = text.lower()
    ids = []
    for entry in r['steps']:
        if entry in SLOTS:
            ids.append(next(step for pattern, step in SLOTS[entry] if re.search(pattern, low)))
        elif entry.startswith('?'):
            if re.search(OPTIONAL.get(entry[1:], r'^$'), low):
                ids.append(entry[1:])
        else:
            ids.append(entry)
    if u['recipe'] == 'basics':
        ids = complete_basics(ids)
    if not ids:
        ids = ['gray']
    kind = u['kind']
    have = set(START[kind.replace('video+', '')]) | {'stages', 'report'}
    for sid in ids:
        missing = [v for v in STEPS[sid]['needs'] if v not in have]
        if missing:
            raise ValueError(f"step '{sid}' needs {missing}, not made by an earlier step")
        have |= set(STEPS[sid]['makes'])
    params = {}
    for sid in ids:
        for name, (default, meaning) in STEPS[sid]['params'].items():
            params.setdefault(name, [default, meaning])
    for name, value in list(r['params'].items()) + list(u['params'].items()):
        if name in params:
            params[name][0] = value
    return dict(u, title=r['title'], goal=r['goal'], steps=ids, params=params, topics=r['topics'])


def complete_basics(ids):
    """Basic operations need gray before histogram; keep the requested order otherwise."""
    if 'histogram_plot' in ids and 'gray' not in ids:
        ids.insert(ids.index('histogram_plot'), 'gray')
    return ids


# ---------------------------------------------------------------- 3a. markdown plan
def ref_name(ref):
    src, _, page = ref.partition(':')
    if src == 'supplement':
        return f'KB supplement ({page})'
    if src == 'cv_fundamentals':
        return 'CV fundamentals, not from the manual'
    if src == 'opencv_docs':
        return f'OpenCV docs: {page}'
    if src == 'opencv_introspection':
        return 'read from installed OpenCV'
    name = ' '.join(w.capitalize() for w in src.split('_'))
    return f"{name} {page.replace('p', 'p.')}" if page else name


def pitfalls(topic_ids):
    seen, out, refs = set(), [], []
    for tid in topic_ids:
        t = TOPICS.get(tid)
        if not t:
            continue
        first = re.split(r'(?<=[.!?])\s+', t['pitfalls'].strip())[:2]
        for sentence in first:
            if sentence and sentence not in seen:
                seen.add(sentence); out.append(sentence)
        refs += [ref_name(r) for r in t['source_refs'] if ref_name(r) not in refs]
    return out, refs


FACTS = [f for f in map(json.loads, (KB / 'records' / 'facts.jsonl').read_text(encoding='utf-8').splitlines()) if f]
TASKS = [t for t in map(json.loads, (KB / 'records' / 'tasks.jsonl').read_text(encoding='utf-8').splitlines()) if t]
_TASK_INDEX = None


def closest_task(text, recipe):
    """The worked lab task (KB tasks.jsonl + solutions/*.py) of this recipe that is most similar to the request.

    Single-task recipes return their task; multi-task recipes pick by TF-IDF similarity, and the generic
    'basics' recipe needs a real word match (score >= 0.25) so an unrelated Lab 01 task is not attached.
    """
    global _TASK_INDEX
    import mcq_answer as M
    if _TASK_INDEX is None:
        _TASK_INDEX = M.Index([(t['id'], t['title'] + '. ' + ' '.join(t['solution_steps']) + ' ' + ' '.join(t['topic_ids']))
                               for t in TASKS])
    candidates = RECIPE_TASKS.get(recipe, [])
    if not candidates:
        return None
    sims = _TASK_INDEX.matrix @ _TASK_INDEX.vector(text)
    score, tid = max((float(sims[_TASK_INDEX.ids.index(t)]), t) for t in candidates)
    if recipe == 'basics' and score < 0.25:
        return None
    task = next(t for t in TASKS if t['id'] == tid)
    return dict(task, score=round(score, 2), source=solution_source(task))


def solution_source(task):
    """Source code of the task's entry point (function or class) from the KB solutions folder."""
    import ast
    path = KB / task['implementation']
    name = task['entry_point'].split('.', 1)[1]
    src = path.read_text(encoding='utf-8')
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(src, node)
    return ''


def manual_facts(sid, k=2):
    """The k course-manual facts that best support a step: same topic or same OpenCV function, most shared words."""
    import mcq_answer as M
    s = STEPS[sid]
    funcs = set(re.findall(r'cv\.([A-Za-z]\w+)\(', s['code']))
    consts = set(re.findall(r'cv\.([A-Z][A-Z_0-9]{2,})', s['code'] + ' ' + str(s['params'])))
    words = set(M.tokens(s['title'] + ' ' + s['why'] + ' ' + ' '.join(funcs)))
    scored = []
    for f in FACTS:
        if not f['source_refs'][0].startswith(('lab_', 'supplement')):
            continue                                   # manual-grounded facts only
        named = any(re.search(r'\b' + fn + r'\(', f['statement']) for fn in funcs)   # "cv2.line(" not "straight line"
        if f['topic_id'] not in s['topics'] and not named:
            continue
        fact_consts = set(re.findall(r'cv2?\.([A-Z][A-Z_0-9]{2,})', f['statement']))
        if fact_consts and not fact_consts & consts:
            continue                                   # same function, different mode (BGR2RGB vs BGR2GRAY)
        overlap = len(words & set(M.tokens(f['statement'])))
        scored.append((overlap + 3 * named, f))
    scored.sort(key=lambda p: -p[0])
    return [f for score, f in scored[:k] if score >= 3]


def fmt_value(v):
    return v if isinstance(v, str) and v.startswith('cv.') else repr(v)


def load_step(plan):
    kind = plan['kind']
    if kind.startswith('video'):
        return ('Open the video and process it frame by frame',
                'A video is a sequence of images. Each frame goes through the same steps; check isOpened() and stop when '
                'read() returns False. Pass 0 instead of a file name for a webcam.',
                "cap = cv.VideoCapture(source)            # file name, or 0 for the webcam\n"
                "if not cap.isOpened():\n    raise FileNotFoundError(source)\n"
                "while True:\n    ok, frame = cap.read()\n    if not ok:\n        break\n"
                f"    stages, report = run({run_args(plan, 'frame')})", ['video'])
    if kind == 'pair':
        return ('Load the reference image and the scene image',
                'imread returns None (no exception) for a wrong path, so check it before using the image.',
                "ref = cv.imread('reference.jpg')\nimg = cv.imread('scene.jpg')\n"
                "if ref is None or img is None:\n    raise FileNotFoundError('check the image paths')", ['io'])
    if kind == 'images':
        return ('Load the overlapping images in left-to-right order',
                'Neighbouring images must overlap (about 30% or more) or there is nothing to match.',
                "imgs = [cv.imread(p) for p in paths]\nif any(i is None for i in imgs) or len(imgs) < 2:\n"
                "    raise FileNotFoundError('need at least two readable images')", ['io'])
    if kind == 'signal':
        return ('Load the sensor readings',
                'One reading per line (CSV). The wavelet steps need a 1-D float array.',
                "signal = np.loadtxt('sensor.csv', delimiter=',')", ['wavelets'])
    return ('Load the image and check it',
            'imread returns None (no exception) for a wrong path, so check it before using the image. '
            'Color images load as BGR.',
            "img = cv.imread('input.jpg')\nif img is None:\n    raise FileNotFoundError('input.jpg')", ['io'])


def show_step(plan):
    if plan['kind'] == 'signal':
        return ('Plot the signal, the denoised curve and the anomalies',
                'Matplotlib shows the original readings, the wavelet-denoised trend and red dots at anomalies.',
                "plt.plot(signal, lw=0.8, label='sensor')\nplt.plot(denoised, label='denoised')\n"
                "plt.scatter(anomalies, signal[anomalies], c='r', label='anomaly')\nplt.legend(); plt.show()", ['plotting'])
    if plan['kind'].startswith('video'):
        return ('Write the annotated frames to an output video',
                'VideoWriter needs the codec, fps and the exact frame size; release both capture and writer at the end.',
                "writer.write(stages.get('Result', frame))   # inside the loop\n"
                "cap.release(); writer.release()", ['video'])
    return ('Show every stage side by side and save the result',
            'Matplotlib expects RGB, so BGR images are converted before plotting; grayscale uses cmap="gray". '
            'The report (counts, states, thresholds) is printed.',
            "for name, value in report.items():\n    print(f'{name}: {value}')\nshow(img, stages)\n"
            "cv.imwrite('result.png', stages['Result'])", ['display', 'plotting'])


def run_args(plan, frame='img'):
    kind = plan['kind'].replace('video+', '')
    return {'image': frame, 'pair': f'ref, {frame}', 'video': f'{frame}, state'}.get(kind, frame)


def to_markdown(plan):
    lines = [f"# Plan: {plan['title']}", '', '> ' + plan['text'].strip().replace('\n', '\n> '), '',
             '## What the task needs', '',
             f"- **Goal:** {plan['goal']}",
             f"- **Input:** {describe_kind(plan['kind'])}",
             f"- **Method:** {plan['title']} (chosen by {plan['via']}: {', '.join(repr(c) for c in plan['cues']) or 'default'})"]
    if plan['alternatives']:
        lines.append('- **Also considered:** ' + ', '.join(f"{RECIPES[a]['title']} ({s})" for a, s in plan['alternatives']))
    limits = [RECIPES[plan['recipe']]['limits']] if RECIPES[plan['recipe']].get('limits') else pitfalls(plan['topics'])[0]
    if limits:
        lines.append('- **Limits to state in your answer:** ' + ' '.join(limits))
    steps = all_steps(plan)
    sids = {i: sid for i, sid in enumerate(plan['steps'], 2)}
    lines += ['', '## Steps at a glance', '']
    lines += [f'{i}. {title}' for i, (title, *_rest) in enumerate(steps, 1)]
    shown = set()                                     # each warning / fact appears once per plan
    for i, (title, why, code, topics, params) in enumerate(steps, 1):
        lines += ['', f'## Step {i}: {title}', '', why]
        if params:
            lines += ['', 'Parameters:'] + [f'- `{n} = {fmt_value(plan["params"][n][0])}`: {plan["params"][n][1]}' for n in params]
        lines += ['', '```python', code, '```']
        notes, srcs = pitfalls(topics)
        if i in sids and STEPS[sids[i]]['watch']:
            notes = [STEPS[sids[i]]['watch']]
        notes = [n for n in notes if n not in shown]
        shown |= set(notes)
        if notes:
            lines += ['', '**Watch out:** ' + ' '.join(notes)]
        facts = [f for f in manual_facts(sids[i]) if f['id'] not in shown] if i in sids else []
        shown |= {f['id'] for f in facts}
        for f in facts:
            note = f" (Note: {f['note']})" if f['note'] else ''
            lines += ['', f"**From the course:** {f['statement']}{note} *({ref_name(f['source_refs'][0])})*"]
        if srcs:
            lines += ['', '*Source: ' + '; '.join(srcs[:4]) + '*']
    task = closest_task(plan["text"], plan["recipe"])
    if task:
        lines += ['', f"## Closest worked lab task: {task['title']} ({task['id']})", '',
                  f"The knowledge base has a tested solution for a similar lab task (match {task['score']}). "
                  f"Its steps:", '']
        lines += [f'{n}. {s}' for n, s in enumerate(task['solution_steps'], 1)]
        lines += ['', f"Limits: {task['assumptions_and_limits']}", '',
                  f"Code (`{task['implementation']}`, `{task['entry_point']}`; helpers come from `solutions/cv_core.py`):",
                  '', '```python', task['source'].rstrip(), '```']
    lines += ['', '## Full script', '', 'Every step above, in order, as one runnable file '
              '(tune the PARAMETERS block for your images):', '', '```python', to_script(plan).rstrip(), '```', '']
    return '\n'.join(lines)


def describe_kind(kind):
    base = {'image': 'one image', 'pair': 'a reference image and a scene image', 'images': 'several overlapping images',
            'video': 'a video (or webcam)', 'signal': 'a 1-D sensor signal'}
    return base[kind.replace('video+', '')] + (' processed frame by frame from a video or webcam' if kind.startswith('video+') else '')


def all_steps(plan):
    """[(title, why, code, topics, param names)] including the load and show steps."""
    out = [load_step(plan) + ([],)]
    for sid in plan['steps']:
        s = STEPS[sid]
        code = s['code'] if not s['helpers'] else s['code'] + '\n\n# helper used above (defined once, outside run):\n' + s['helpers']
        out.append((s['title'], s['why'], code, s['topics'], list(s['params'])))
    out.append(show_step(plan) + ([],))
    return out


# ---------------------------------------------------------------- 3b. runnable script
def to_script(plan):
    kind = plan['kind']
    base = kind.replace('video+', '')
    body = []
    for i, sid in enumerate(plan['steps'], 2):
        body += [f'# Step {i}: {STEPS[sid]["title"]}', STEPS[sid]['code'], '']
    helpers, seen = [], set()
    for sid in plan['steps']:
        h = STEPS[sid]['helpers']
        if h and h not in seen:
            seen.add(h); helpers.append(h)
    params = '\n'.join(f'{n} = {fmt_value(v)}{" " * max(1, 24 - len(n) - len(fmt_value(v)))}# {m}' for n, (v, m) in plan['params'].items())
    returns = ', '.join(v for v in ('denoised', 'anomalies') if base == 'signal')
    task = textwrap.fill(plan['text'].strip().replace('"""', "'''"), 100, subsequent_indent='      ')
    head = (f'"""{plan["title"]}.\n\nTask: {task}\n\n'
            'Generated offline by the CV planner from the course knowledge base. Step 1 (loading) is in main().\n"""\n'
            'import sys\n\nimport cv2 as cv\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n'
            '# ---- PARAMETERS: tune these for your images ----\n' + params + '\n')
    run = (f'\n\ndef run({SIGNATURE[base]}):\n    """Run the plan on one input; returns (stages to display, report of numbers)."""\n'
           '    stages, report = {}, {}\n' + textwrap.indent('\n'.join(body).rstrip(), '    ') + '\n'
           + (f'    return stages, report, {returns}\n' if returns else '    return stages, report\n'))
    return head + ('\n\n' + '\n\n\n'.join(helpers) + '\n' if helpers else '') + run + '\n\n' + SHOW + '\n\n' + main_code(plan) + \
        "\n\nif __name__ == '__main__':\n    main()\n"


SHOW = '''def show(original, stages):
    """Original plus every stage in one matplotlib figure (BGR converted to RGB)."""
    items = [('Original', original)] + [(k, v) for k, v in stages.items() if isinstance(v, np.ndarray) and v.ndim in (2, 3)]
    cols = min(3, len(items)); rows = (len(items) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (name, image) in enumerate(items, 1):
        plt.subplot(rows, cols, i)
        if image.ndim == 3:
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title(name); plt.axis('off')
    plt.tight_layout(); plt.show()'''


def main_code(plan):
    kind = plan['kind']
    base = kind.replace('video+', '')
    if kind.startswith('video'):
        ref = ("    ref = cv.imread(sys.argv[2] if len(sys.argv) > 2 else 'reference.jpg')\n"
               "    if ref is None:\n        raise FileNotFoundError('reference image')\n") if base == 'pair' else ''
        state = '    state = {}\n' if base == 'video' else ''
        return ('def main():\n'
                "    source = sys.argv[1] if len(sys.argv) > 1 else 'input.mp4'\n" + ref +
                '    cap = cv.VideoCapture(int(source) if source.isdigit() else source)   # 0 = webcam\n'
                '    if not cap.isOpened():\n        raise FileNotFoundError(source)\n'
                '    fps = cap.get(cv.CAP_PROP_FPS) or 25\n' + state +
                '    writer, n = None, 0\n'
                '    while True:\n'
                '        ok, frame = cap.read()\n'
                '        if not ok:\n            break\n'
                f'        stages, report = run({run_args(plan, "frame")})\n'
                "        out = stages.get('Result', frame)\n"
                '        if writer is None:\n'
                "            writer = cv.VideoWriter('result.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, out.shape[1::-1])\n"
                '        writer.write(out)\n'
                '        n += 1\n'
                '        if n % 25 == 1 or report.get(\'alert\'):\n'
                "            print(f'frame {n}:', {k: v for k, v in report.items() if not isinstance(v, (list, np.ndarray))})\n"
                '    cap.release()\n'
                '    if writer is not None:\n        writer.release()\n'
                "    print(f'{n} frames written to result.mp4')")
    if base == 'pair':
        return ('def main():\n'
                "    ref = cv.imread(sys.argv[1] if len(sys.argv) > 1 else 'reference.jpg')\n"
                "    img = cv.imread(sys.argv[2] if len(sys.argv) > 2 else 'scene.jpg')\n"
                "    if ref is None or img is None:\n        raise FileNotFoundError('check the reference and scene paths')\n"
                '    stages, report = run(ref, img)\n'
                "    for name, value in report.items():\n        print(f'{name}: {value}')\n"
                "    show(img, stages)\n    cv.imwrite('result.png', stages['Result'])")
    if base == 'images':
        return ('def main():\n'
                "    paths = sys.argv[1:] or ['left.jpg', 'right.jpg']\n"
                '    imgs = [cv.imread(p) for p in paths]\n'
                "    if len(imgs) < 2 or any(i is None for i in imgs):\n        raise FileNotFoundError('need at least two readable images')\n"
                '    stages, report = run(imgs)\n'
                "    for name, value in report.items():\n        print(f'{name}: {value}')\n"
                "    show(imgs[0], stages)\n    cv.imwrite('panorama.png', stages['Panorama'])")
    if base == 'signal':
        return ('def main():\n'
                "    signal = np.loadtxt(sys.argv[1] if len(sys.argv) > 1 else 'sensor.csv', delimiter=',').ravel()\n"
                '    stages, report, denoised, anomalies = run(signal)\n'
                "    print('anomaly indices:', report['anomalies'])\n"
                "    plt.figure(figsize=(12, 4))\n    plt.plot(signal, lw=0.8, label='sensor')\n"
                "    plt.plot(denoised, label='denoised')\n"
                "    plt.scatter(anomalies, signal[anomalies], c='r', zorder=3, label='anomaly')\n"
                "    plt.legend(); plt.title('Sensor data, denoised signal and anomalies'); plt.show()")
    save = "    if 'Result' in stages:\n        cv.imwrite('result.png', stages['Result'])"
    return ('def main():\n'
            "    path = sys.argv[1] if len(sys.argv) > 1 else 'input.jpg'\n"
            '    img = cv.imread(path)\n'
            '    if img is None:\n        raise FileNotFoundError(path)\n'
            '    stages, report = run(img)\n'
            "    for name, value in report.items():\n"
            "        print(f'{name}: {value}' if not isinstance(value, np.ndarray) else f'{name}: array {value.shape}')\n"
            '    show(img, stages)\n' + save)


def plan(text):
    return build(text)


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in argv if a.startswith('--') and '=' in a)
    text = Path(args[0]).read_text(encoding='utf-8') if args and Path(args[0]).is_file() else ' '.join(args)
    p = build(text)
    md = to_markdown(p)
    if 'out' in opts:
        Path(opts['out']).write_text(md, encoding='utf-8')
        print('plan ->', opts['out'])
    if 'script' in opts:
        Path(opts['script']).write_text(to_script(p), encoding='utf-8')
        print('script ->', opts['script'])
    if not opts:
        print(md)


if __name__ == '__main__':
    main(sys.argv[1:])
