"""Write a plan (markdown) and a runnable script for every course lab task into outputs/cv_knowledge_base/plans/.

The task list is work/eval/plan_dev.txt. PLANS.md indexes them.
"""
from pathlib import Path
import re, sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planner  # noqa: E402

OUT = HERE.parent / 'outputs' / 'cv_knowledge_base' / 'plans'


def slug(text, used):
    s = '-'.join(re.findall(r'[a-z0-9]+', text.lower())[:6])
    base, n = s, 2
    while s in used:
        s, n = f'{base}-{n}', n + 1
    used.add(s)
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [l.split('|', 1)[1] for l in (HERE / 'eval' / 'plan_dev.txt').read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.startswith('#')]
    used, index = set(), ['# Plans for the course lab tasks', '',
                          'Generated offline by `work/planner.py` from the knowledge base. Each plan has the steps, '
                          'why each step is there, its parameters, code, warnings and manual references, plus a full '
                          'script (`.py`, same code) that was checked by `work/plan_selftest.py`.', '',
                          '| Task | Method | Steps | Files |', '|---|---|---|---|']
    for text in rows:
        p = planner.build(text)
        name = slug(text, used)
        (OUT / f'{name}.md').write_text(planner.to_markdown(p), encoding='utf-8')
        (OUT / f'{name}.py').write_text(planner.to_script(p), encoding='utf-8')
        short = text if len(text) < 90 else text[:87] + '...'
        index.append(f"| {short} | {p['title']} | {len(p['steps']) + 2} | [plan]({name}.md), [script]({name}.py) |")
    (OUT / 'PLANS.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    print(f'{len(rows)} plans written to {OUT}')


if __name__ == '__main__':
    main()
