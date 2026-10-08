"""Ask a computer-vision question, get a written answer. Offline, nothing is executed, no image needed.

    python work/ask.py "Why do we blur before Canny edge detection?"
    python work/ask.py "Detect computer screens in a lab using Hough lines and report missing ones"
    python work/ask.py question.txt --out=answer.md        # long question from a file, answer saved to a file
    python work/ask.py "..." --detail                      # task questions: full per-step plan instead of the short form

Two kinds of answer:
  conceptual question (what / why / how does / explain / difference ...)
      -> explanation from the knowledge base, key points from the course manuals with page numbers,
         common mistakes and a short code example
  task question (detect / count / implement / "you are working on ..." ...)
      -> approach, numbered steps with the reason for each, and the complete code (from planner.py)
"""
from pathlib import Path
import ast, re, sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planner  # noqa: E402
import mcq_answer as M  # noqa: E402

CONCEPT = re.compile(r'^\s*(what|why|how (does|do|is|are|can|would)|explain|define|describe|state|list|mention|'
                     r'what\'s|difference|differentiate|distinguish between|compare (the )?(concept|idea)|when (should|would|do|is)|'
                     r'which|is it|can we|does|do we|name)\b', re.I)
TASK = re.compile(r'\b(implement|write (a |the )?(code|program|function|script)|you are (working|tasked|responsible)|'
                  r'your (goal|task)|build a|create a|develop a|apply|detect|count|segment|stitch|recogni[sz]e|'
                  r'isolate|extract|rotate|crop|draw|load|monitor|track|find (all|each|every|the)|enhance|fuse)\b', re.I)


def is_conceptual(q):
    """Conceptual if it opens like a theory question; a task if it asks for something to be built or done."""
    if CONCEPT.search(q) and not re.search(r'\b(implement|write (a |the )?(code|program|script)|you are working)\b', q, re.I):
        return True
    return not TASK.search(q) and q.strip().endswith('?')


# ---------------------------------------------------------------- conceptual answers
# Same idea, different words: the question says "blur", the manual says "Gaussian smoothing reduces noise".
SYNONYMS = [
    'blur blurring smooth smoothing gaussian denoise noise',
    'threshold thresholding binarize binary mask',
    'edge edges boundary boundaries contour',
    'keypoint keypoints feature features interest point descriptor',
    'match matching correspondence correspondences pair',
    'rotate rotation rotated angle',
    'warp warping transform transformation mapping',
    'segment segmentation partition regions',
    'count counting number how many',
    'float32 float floating type dtype',
    'outlier outliers wrong matches inlier inliers',
    'scale scaling resize size',
    'histogram distribution intensities',
    'color colour hue hsv',
    'gray grayscale grey single channel intensity',
    'lighting light illumination brightness',
]
_SYN = {w: line for line in SYNONYMS for w in line.split()}
REASON = re.compile(r'\b(to (prevent|avoid|reduce|remove|keep|make|get)|because|so that|reduces?|prevents?|removes?|'
                    r'avoids?|otherwise|needs?|requires?|must|purpose|used (to|for))\b', re.I)
TRIVIA = re.compile(r'\b(developed by|invented|proposed by|in (19|20)\d\d)\b', re.I)


def expand(q):
    """Add CV synonyms of the question's words (once each) so paraphrased facts are found."""
    words = re.findall(r'[a-z0-9]+', q.lower())
    extra = {_SYN[w] for w in words if w in _SYN}
    return q + ' ' + ' '.join(extra)


RERANKER_DIR = HERE.parent / 'models' / 'ms-marco-MiniLM-L-6-v2'


class Reranker:
    """Question-passage relevance cross-encoder (MiniLM trained on MS MARCO, 8-bit ONNX, ~23 MB, CPU)."""

    def __init__(self, model_dir=RERANKER_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
        self.tok.enable_truncation(256)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_dir / 'model_quantized.onnx'), opts, providers=['CPUExecutionProvider'])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def scores(self, question, passages):
        import numpy as np
        enc = self.tok.encode_batch([(question, p) for p in passages])
        feed = {'input_ids': np.array([e.ids for e in enc], np.int64),
                'attention_mask': np.array([e.attention_mask for e in enc], np.int64),
                'token_type_ids': np.array([e.type_ids for e in enc], np.int64)}
        return self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][:, 0]


_RERANKER = None
TYPE_WEIGHT = 4.0   # bonus for a fact that matches the question type (why -> reason, difference -> contrast, how -> mechanism)
MECHANISM = re.compile(r'\b(votes?|computes?|compares?|repeats?|assigns?|represents?|maps?|keeps?|splits?|divides?|'
                       r'counts?|accumulat\w*|tries|picks?|by (comparing|computing|voting))\b', re.I)
LEX_WEIGHT = 3.0   # word-match weight next to the cross-encoder score (set on work/eval/concept_dev.txt)


def reranker():
    global _RERANKER
    if _RERANKER is None and RERANKER_DIR.exists():
        _RERANKER = Reranker()
    return _RERANKER


_IDX = {}


def _index(kind):
    if kind not in _IDX:
        if kind == 'topics':
            rows = [(t['id'], ' '.join([t['title'], ' '.join(t['aliases']), t['answer'], t['when_to_use'], t['pitfalls']]))
                    for t in planner.TOPICS.values()]
        else:
            rows = [(f['id'], f['statement'] + ' ' + f['note']) for f in planner.FACTS]
        _IDX[kind] = M.Index(rows)
    return _IDX[kind]


# How the user wants the answer: "in one line", "briefly", "in 5 points", "in detail", "with code", "only code".
STYLE = [
    ('line', re.compile(r'\b(in |answer in )?(one|1|a single|single) (line|sentence)\b|\bone[- ]liner\b', re.I)),
    ('code_only', re.compile(r'\b(only|just) (the )?code\b|\bcode only\b', re.I)),
    ('points', re.compile(r'\bin (\d+|two|three|four|five|six) (points|bullets|bullet points)\b|\bin points\b|\bas (a )?list\b', re.I)),
    ('short', re.compile(r'\b(briefly|in brief|short answer|in short|quick(ly)?|in (two|three|2|3) (lines|sentences))\b', re.I)),
    ('detail', re.compile(r'\b(in detail|detailed|elaborate|explain fully|long answer|step by step)\b', re.I)),
]
WITH_CODE = re.compile(r'\b(with (a |the )?code|code example|show (the )?code|write (the )?code|with (an )?example)\b', re.I)
NUMBERS = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}


def parse_style(q):
    """(style, number of points, wants code, question without the formatting instructions)."""
    style, n, clean = 'normal', 4, q
    for name, pat in STYLE:
        m = pat.search(q)
        if m and style == 'normal':
            style = name
            if name == 'points':
                d = re.search(r'(\d+|two|three|four|five|six) (points|bullets)', m.group(0), re.I)
                n = int(NUMBERS.get(d.group(1).lower(), d.group(1))) if d else 4
        clean = pat.sub('', clean)
    wants_code = bool(WITH_CODE.search(q)) or style == 'code_only'
    clean = WITH_CODE.sub('', clean).strip()
    clean = re.sub(r'[\s,;:-]*\b(answer|explain|tell me|please)\b[\s,.;:!?-]*$', '', clean, flags=re.I)
    clean = re.sub(r'[\s,;:-]+$', '', clean).strip()
    clean = re.sub(r'\?\s*[.?]*$', '?', clean)
    return style, n, wants_code, clean or q


def _words(text):
    return {w for w in M.tokens(text) if '__' not in w}


def similar(a, b, limit=0.5):
    """True if two statements say mostly the same thing (share most of the smaller one's words)."""
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) > limit


def retrieve(q, n_facts=8):
    """(topics, ranked distinct facts) for a conceptual question."""
    u = planner.understand(q)
    hinted = set(planner.RECIPES[u['recipe']]['topics']) if u['cues'] else set()
    ti = _index('topics')
    tsim = ti.matrix @ ti.vector(q)
    ranked = sorted(range(len(ti.ids)), key=lambda i: -(tsim[i] + 0.1 * (ti.ids[i] in hinted)))
    topics = [planner.TOPICS[ti.ids[i]] for i in ranked[:2]]
    if tsim[ranked[1]] < 0.6 * tsim[ranked[0]]:
        topics = topics[:1]                                    # second topic only if nearly as relevant
    chosen = {t['id'] for t in topics}
    fi = _index('facts')
    fsim = fi.matrix @ fi.vector(expand(q))
    by_id = {f['id']: f for f in planner.FACTS}
    why = bool(re.match(r'\s*(why|what is the (purpose|reason|role)|what does .* do)', q, re.I))
    trivia_ok = bool(re.search(r'\b(who|when|year|history)\b', q, re.I))

    def fact_score(i):
        f = by_id[fi.ids[i]]
        s = fsim[i] + 0.08 * (f['topic_id'] in chosen | hinted)
        s += 0.06 * (why and bool(REASON.search(f['statement'])))       # "why" wants a reason
        s -= 0.15 * (not trivia_ok and bool(TRIVIA.search(f['statement'])))
        return s
    franked = sorted(range(len(fi.ids)), key=lambda i: -fact_score(i))
    pool = [i for i in franked[:40] if fsim[i] > 0.05]
    rr = reranker()
    if rr and pool:
        # Word search finds candidates; the cross-encoder orders them by how well they answer the question.
        ce = rr.scores(q, [by_id[fi.ids[i]]['statement'] for i in pool])
        # Exact two-word phrases from the question ("ratio test", "distance transform") mark the fact it is about.
        q_pairs = {w for w in M.tokens(q) if '__' in w}
        diff = bool(re.search(r'\b(difference|differ|compare|versus|vs)\b', q, re.I))
        how = bool(re.match(r'\s*(how (does|do|is|are)|explain how)\b', q, re.I))
        bonus = []
        for i in pool:
            st = by_id[fi.ids[i]]['statement']
            b = 1.5 * min(2, len(q_pairs & {w for w in M.tokens(st) if '__' in w}))
            b += LEX_WEIGHT * fsim[i]                                # keep the word-match evidence, not CE alone
            b += TYPE_WEIGHT * (why and bool(REASON.search(st)))              # "why" wants a reason
            b += TYPE_WEIGHT * (diff and bool(re.search(r'\b(while|whereas|versus|unlike)\b|;', st, re.I)))   # a contrast
            b += TYPE_WEIGHT * (how and bool(MECHANISM.search(st)))           # "how does it work" wants the mechanism
            bonus.append(b)
        ce = ce + bonus
        order = sorted(range(len(pool)), key=lambda j: -(ce[j] - 3.0 * (not trivia_ok and bool(TRIVIA.search(by_id[fi.ids[pool[j]]]['statement'])))))
        ranked_facts = [by_id[fi.ids[pool[j]]] for j in order if ce[j] > -6]
    else:
        ranked_facts = [by_id[fi.ids[i]] for i in franked if fsim[i] > 0.12]
    facts = []
    for f in ranked_facts:                                     # drop near-duplicates of better-ranked facts
        if not any(similar(f['statement'], g['statement']) for g in facts):
            facts.append(f)
        if len(facts) == n_facts:
            break
    return topics, facts


PREAMBLE = {
    'image': "image = cv.imread('input.jpg')          # your image\nif image is None:\n    raise FileNotFoundError('input.jpg')",
    'g': 'g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # grayscale',
    'gray': 'gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)',
}


def runnable(code):
    """Add the load/grayscale lines a topic snippet assumes, so the snippet runs on its own."""
    import builtins
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    stored = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    stored |= {a.arg for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.Lambda)) for a in n.args.args}
    stored |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    stored |= {a.asname or a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = loaded - stored - set(dir(builtins)) - {'cv', 'np', 'plt', 'math'}
    head = []
    if missing & {'image', 'img', 'g', 'gray'}:
        head.append(PREAMBLE['image'])
        if 'img' in missing:
            head.append('img = image')
    for name in ('g', 'gray'):
        if name in missing:
            head.append(PREAMBLE[name])
    rest = sorted(missing - {'image', 'img', 'g', 'gray'})
    if rest:
        head.append('# define first: ' + ', '.join(rest))
    return '\n'.join(head + ([''] if head else []) + [code.strip()])


def code_block(topic):
    code = (topic.get('setup', '') + '\n' if topic.get('setup') else '') + topic['code']
    return ['```python', 'import cv2 as cv', 'import numpy as np', '', runnable(code), '```']


def conceptual(q):
    style, n_points, wants_code, clean = parse_style(q)
    topics, facts = retrieve(clean)
    cite = lambda f: f"*[{planner.ref_name(f['source_refs'][0])}]*"
    note = lambda f: f" (More precisely: {f['note']})" if f['note'] else ''
    top = facts[0] if facts else None

    if style == 'line':                                       # one sentence, nothing else
        if not top:
            return re.split(r'(?<=[.!?])\s+', topics[0]['answer'])[0] + '\n'
        return f"{top['statement']} {cite(top)}\n"
    if style == 'code_only':
        return '\n'.join(code_block(topics[0])) + '\n' if topics[0].get('code') else 'No code example for this topic.\n'
    if style == 'points':
        lines = [f'# {clean}', ''] + [f"{i}. {f['statement']}{note(f)} {cite(f)}" for i, f in enumerate(facts[:n_points], 1)]
        if wants_code and topics[0].get('code'):
            lines += [''] + code_block(topics[0])
        return '\n'.join(lines) + '\n'

    out = [f'# {clean}', '']
    if top:
        out += [f"**Answer:** {top['statement']}{note(top)} {cite(top)}", '']
    if style == 'short':
        out += [f"- {f['statement']} {cite(f)}" for f in facts[1:3]]
        if wants_code and topics[0].get('code'):
            out += [''] + code_block(topics[0])
        return '\n'.join(out) + '\n'

    # normal / detail
    for t in (topics if style == 'detail' else topics[:1]):
        out += [f"## Explanation{': ' + t['title'] if style == 'detail' else ''}", '', t['answer'], '']
    rest = facts[1:(8 if style == 'detail' else 4)]
    if rest:
        out += ['## Key points', ''] + [f"- {f['statement']}{note(f)} {cite(f)}" for f in rest] + ['']
    pits = [t['pitfalls'] for t in topics if t.get('pitfalls')]
    if pits:
        shown = pits if style == 'detail' else [' '.join(re.split(r'(?<=[.!?])\s+', pits[0])[:2])]
        out += ['## Common mistakes', ''] + [f'- {p}' for p in shown] + ['']
    if (wants_code or style == 'detail' or re.match(r'\s*how (do|to|can|would)', clean, re.I)) and topics[0].get('code'):
        out += [f"## Code: {topics[0]['title']}", ''] + code_block(topics[0]) + ['']
    refs = sorted({planner.ref_name(r) for t in topics for r in t['source_refs']})
    out += ['*Sources: ' + '; '.join(refs) + '*']
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------- task answers
def task_answer(q, detail=False):
    p = planner.build(q)
    if detail:
        return planner.to_markdown(p)
    steps = planner.all_steps(p)
    out = [f'# {p["title"]}', '', '> ' + q.strip().replace('\n', '\n> '), '', '## Approach', '',
           f"Goal: {p['goal']}. Input: {planner.describe_kind(p['kind'])}. Method: {p['title']}.", '']
    limits = planner.RECIPES[p['recipe']].get('limits') or ' '.join(planner.pitfalls(p['topics'])[0])
    out += ['## Steps', '']
    for i, (title, why, *_rest) in enumerate(steps, 1):
        first = re.split(r'(?<=[.!?])\s+', why.strip())[0]
        out.append(f'{i}. **{title}.** {first}')
    out += ['', '## Code', '', '```python', planner.to_script(p).rstrip(), '```', '']
    if limits:
        out += ['## Limitations', '', limits, '']
    return '\n'.join(out)


def answer(q, detail=False):
    return conceptual(q) if is_conceptual(q) else task_answer(q, detail)


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], True) for a in argv if a.startswith('--'))
    q = Path(args[0]).read_text(encoding='utf-8') if args and Path(args[0]).is_file() else ' '.join(args)
    text = answer(q, detail=bool(opts.get('detail')))
    if 'out' in opts:
        Path(opts['out']).write_text(text, encoding='utf-8')
        print('answer ->', opts['out'])
    else:
        print(text)


if __name__ == '__main__':
    main(sys.argv[1:])
