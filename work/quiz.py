"""Answer multiple-choice questions from the knowledge base and write an answer sheet.

The engine only ever sees question + options. Answer keys (if present) are removed before
answering and used afterwards, only to mark the sheet.

Input formats:
  * plain text as pasted: "12. Question?" then "A) ..." / "A. ..." option lines; lines with
    "Correct Answer" and the check mark are stripped from what the engine sees.
  * key files from work/eval (section|question|A|B|C|D|key).

Usage:
    python work/quiz.py questions.txt [--out=sheet.md] [--model=nli-deberta-v3-small]
"""
from pathlib import Path
import re, sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OPTION = re.compile(r'^\s*([A-Da-d])[\).:]\s*(.+?)\s*$')
KEYLINE = re.compile(r'correct answer\s*:?\s*\**\s*([A-D])\b', re.I)
CHECK = '✅'


def parse_text(text):
    """Return list of dict(question, options, key or None) from pasted text."""
    items, cur = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = KEYLINE.search(line)
        if m:
            if cur is not None:
                cur['key'] = m.group(1).upper()
            continue
        marked = CHECK in line
        line = line.replace(CHECK, '').strip().strip('*').strip()
        if not line or line.lower().startswith(('correct answer', 'instructions')):
            continue
        o = OPTION.match(line)
        if o and cur is not None and len(cur['options']) < 4:
            cur['options'].append(o.group(2).strip().strip('*').strip())
            if marked:
                cur['key'] = o.group(1).upper()
            continue
        if line.lower().startswith(('section', 'lab 0', 'mcqs from', 'below are')) and not line.endswith(('?', ':')):
            continue
        q = re.sub(r'^\s*\d+[\.\)]\s*', '', line)
        if cur is None or len(cur['options']) >= 2:
            cur = dict(question=q, options=[], key=None)
            items.append(cur)
        else:
            cur['question'] += ' ' + q
    return [i for i in items if len(i['options']) >= 2]


INLINE_OPTION = re.compile(r'\s+(?=\(?[A-Da-d][\).:]\s)')


def split_inline(text):
    """'Question? A) x B) y C) z D) w' typed on one line -> one option per line (only if 2+ options are found)."""
    if '\n' in text.strip() or len(INLINE_OPTION.findall(text)) < 2:
        return text
    return INLINE_OPTION.sub('\n', text).replace('\n(', '\n')


def answer_terminal(items, engine=None):
    """Answer parsed MCQs and print a compact result for each to the terminal."""
    from mcq_engine import Engine
    import planner
    engine = engine or Engine()
    graded = correct = 0
    for n, it in enumerate(items, 1):
        r = engine.answer(it['question'], list(it['options']))
        letter = 'ABCD'[r['choice']]
        mark = ''
        if it.get('key'):
            graded += 1
            ok = letter == it['key']
            correct += ok
            mark = '  [correct]' if ok else f"  [WRONG, key is {it['key']}]"
        ref = str(r.get('evidence', '')).split(' ')[-1]
        src = planner.ref_name(ref) if ':' in ref else ''
        print(f"\nQ{n}. {it['question']}")
        print(f"   Answer: {letter}) {it['options'][r['choice']]}{mark}")
        if r.get('fact'):
            print(f"   Because: {r['fact']}" + (f'  [{src}]' if src else ''))
    if graded:
        print(f'\nScore: {correct}/{graded}')
    return engine


def parse_keyfile(text):
    items = []
    for l in text.splitlines():
        if l.strip() and not l.startswith('#'):
            _, q, *opts, key = l.split('|')
            items.append(dict(question=q, options=opts, key=key.strip()))
    return items


def load(path):
    text = Path(path).read_text(encoding='utf-8')
    return parse_keyfile(text) if '|' in text.splitlines()[-1] else parse_text(text)


def main(argv):
    from mcq_engine import Engine
    items = load(argv[0])
    out = Path(next((a.split('=', 1)[1] for a in argv if a.startswith('--out=')), Path(argv[0]).with_suffix('.answers.md')))
    model = next((a.split('=', 1)[1] for a in argv if a.startswith('--model=')), None)
    engine = Engine(model_dir=HERE.parent / 'models' / model) if model else Engine()
    lines = [f'# Answer sheet: {Path(argv[0]).name}', '',
             'The engine saw only the question and options. "Evidence" is the knowledge-base fact it relied on.', '']
    graded = correct = 0
    wrong = []
    for n, it in enumerate(items, 1):
        r = engine.answer(it['question'], [o for o in it['options']])
        letter = 'ABCD'[r['choice']]
        mark = ''
        if it['key']:
            graded += 1
            ok = letter == it['key']
            correct += ok
            mark = ' ✅' if ok else f" ❌ (key {it['key']}: {it['options']['ABCD'.index(it['key'])]})"
            if not ok:
                wrong.append(n)
        lines += [f"**{n}. {it['question']}**", '']
        lines += [f"- {'ABCD'[j]}) {o}" for j, o in enumerate(it['options'])]
        lines += ['', f"Answer: **{letter}) {it['options'][r['choice']]}**{mark}",
                  f"Evidence ({r['layer']}): {r.get('fact', '')} — `{r.get('evidence', '')}`", '']
    summary = f'Graded {correct}/{graded} correct.' if graded else 'No answer key supplied.'
    if wrong:
        summary += ' Wrong: ' + ', '.join(f'Q{n}' for n in wrong)
    lines.insert(2, summary + '\n')
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'{Path(argv[0]).name}: {summary}  -> {out}')


if __name__ == '__main__':
    main(sys.argv[1:])
