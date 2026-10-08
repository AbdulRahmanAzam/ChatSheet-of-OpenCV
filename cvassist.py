"""cvassist: offline OpenCV study assistant. One entry point for every command.

    python cvassist.py help                 # what you can do, with examples
    python cvassist.py <command> --help     # details of one command
"""
from pathlib import Path
import argparse, os, sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'work'
sys.path.insert(0, str(WORK))
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HELP = r"""
cvassist - offline OpenCV / computer-vision study assistant (no internet, no big AI model)

COMMANDS
  ask      Answer a theory question, or plan + write code for a task          (main command)
  mcq      Answer multiple-choice questions: type them, paste them, or give a file
  plan     Full step-by-step plan for a task (every step: why, settings, code, warnings, manual facts)
  search   Look up topics in the knowledge base
  code     EXPERIMENTAL: write new code for a custom "write solve(img) that ..." question
  check    Check that this machine has everything installed
  help     Show this help

EXAMPLES
  cvassist ask "Why do we blur before Canny edge detection?"
  cvassist ask "Why do we blur before Canny edge detection? answer in one line"
  cvassist ask "Explain watershed segmentation in 3 points"
  cvassist ask "How does Otsu thresholding work? with code"
  cvassist ask "Detect lane markings on the road using Hough lines" --out answer.md
  cvassist mcq "Why blur before Canny? A) more contrast B) reduce noise C) grayscale D) thicker edges"
  cvassist mcq                      (then paste one or many MCQs, press Enter twice)
  cvassist mcq mcqs.txt             (answers printed; add --out answers.md for a sheet file)
  cvassist plan "Count the coins using Hough circles" --out plan.md --script solution.py
  cvassist search "ratio test"
  cvassist check

HOW TO PHRASE AN ASK QUESTION
  Theory question  -> starts with what / why / how / explain / define / difference ...
  Task question    -> detect / count / segment / implement / "You are working on ..." ...
  Add to the question to control the answer:
     "in one line"   one sentence          "briefly"      answer + 2 points
     "in 3 points"   numbered list          "in detail"    everything
     "with code"     add runnable code      "only code"    just the code

MCQ FORMAT (when pasting or in a file; one blank line between questions)
  1. Why is the image blurred before applying Canny?
  A) To increase contrast
  B) To reduce noise that would create false edges
  C) To convert it to grayscale
  D) To make edges thicker
  Correct Answer: B          <- optional; if given, the sheet is marked right/wrong

Run  cvassist <command> --help  for the options of one command.
(Without the cvassist.bat shortcut use:  python cvassist.py <command> ...)
"""


def cmd_ask(a):
    import ask
    argv = [a.question] + (['--out=' + a.out] if a.out else []) + (['--detail'] if a.detail else [])
    ask.main(argv)


def read_pasted():
    """Read pasted MCQs from the terminal until an empty line follows a complete question (or Ctrl+Z / Ctrl+D)."""
    print('Paste or type the MCQ(s): question line, then options A) B) C) D) (a "Correct Answer: X" line is optional).\n'
          'Press Enter on an empty line twice to get the answers.\n', file=sys.stderr)
    lines, blanks = [], 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        blanks = blanks + 1 if not line.strip() else 0
        if blanks >= 2 and any(l.strip() for l in lines):
            break
        lines.append(line)
    return '\n'.join(lines)


def cmd_mcq(a):
    import quiz
    source = ' '.join(a.text) if a.text else ''
    if source and Path(source).is_file():                       # a file of questions
        if a.out:
            quiz.main([source, '--out=' + a.out] + (['--model=' + a.model] if a.model else []))
            return
        items = quiz.load(source)
    else:                                                       # typed inline, or pasted
        text = quiz.split_inline(source) if source else read_pasted()
        items = quiz.parse_text(text)
    if not items:
        sys.exit('No question with options found. Format:  Question?  A) ...  B) ...  C) ...  D) ...  (see: cvassist help)')
    print('Loading the answer model ...', file=sys.stderr)
    quiz.answer_terminal(items)


def cmd_plan(a):
    import planner
    argv = [a.task] + (['--out=' + a.out] if a.out else []) + (['--script=' + a.script] if a.script else [])
    planner.main(argv)


def cmd_search(a):
    sys.path.insert(0, str(ROOT / 'outputs' / 'cv_knowledge_base'))
    import search
    results = search.search(a.query, limit=a.limit)
    if not results:
        print('Nothing found. Try other words, or use: cvassist ask "your question"')
    for r in results:
        rec = r['record']
        text = (rec.get('answer') or rec.get('statement') or rec.get('text') or rec.get('summary') or '')[:300]
        print(f"- [{r['kind']}] {r['title']}" + ('' if text.strip() == str(r['title']).strip() else f'\n  {text}') + '\n')


def cmd_code(a):
    import codegen
    if not (codegen.MODELS.exists() and any(codegen.MODELS.glob('*.gguf'))):
        sys.exit('The code model is not installed (models/qwen2.5-coder/*.gguf). See HOW_TO_USE.md, section "Optional".')
    argv = [a.question] + (['--out=' + a.out] if a.out else []) + (['--model=' + a.model] if a.model else [])
    codegen.main(argv)


def cmd_check(a):
    ok = True
    print(f'Python {sys.version.split()[0]}', 'OK' if sys.version_info >= (3, 10) else 'TOO OLD (need 3.10+)')
    for mod, need in [('numpy', ''), ('cv2', '4.x (not 5.x)'), ('matplotlib', ''), ('onnxruntime', ''), ('tokenizers', '')]:
        try:
            m = __import__(mod)
            v = getattr(m, '__version__', '?')
            bad = mod == 'cv2' and not v.startswith('4.')
            print(f'  {mod:12s} {v:10s} {"WRONG VERSION, need " + need if bad else "OK"}')
            ok &= not bad
        except Exception as e:
            print(f'  {mod:12s} MISSING  -> pip install -r requirements.txt'); ok = False
    for name, path, needed_for in [
        ('knowledge base', ROOT / 'outputs' / 'cv_knowledge_base' / 'records' / 'facts.jsonl', 'everything'),
        ('MCQ reader', ROOT / 'models' / 'nli-deberta-v3-base' / 'model_quint8_avx2.onnx', 'mcq'),
        ('answer re-ranker', ROOT / 'models' / 'ms-marco-MiniLM-L-6-v2' / 'model_quantized.onnx', 'ask (better answers)'),
    ]:
        print(f'  {name:18s} {"OK" if path.exists() else "MISSING (" + str(path.relative_to(ROOT)) + ")"}   needed for: {needed_for}')
        ok &= path.exists()
    gguf = list((ROOT / 'models' / 'qwen2.5-coder').glob('*.gguf')) if (ROOT / 'models' / 'qwen2.5-coder').exists() else []
    print(f'  {"code model":18s} {"OK" if gguf else "not installed (optional, only for: code)"}')
    if ok:
        import ask
        print('\nTest answer:', ask.answer('What is HOG? in one line').strip()[:150])
    print('\nAll set.' if ok else '\nFix the items above, then run  cvassist check  again.')


def main(argv=None):
    p = argparse.ArgumentParser(prog='cvassist', add_help=False,
                                description='Offline OpenCV study assistant. Run "cvassist help" for examples.')
    sub = p.add_subparsers(dest='command')

    s = sub.add_parser('ask', help='answer a theory question or plan + code a task',
                       description='Theory question -> written answer from the course knowledge base. '
                                   'Task -> approach, steps and complete code. Add "in one line", "briefly", '
                                   '"in 3 points", "in detail", "with code" or "only code" to the question.')
    s.add_argument('question', help='the question in quotes, or a .txt file containing it')
    s.add_argument('--out', help='save the answer to this file (e.g. answer.md) instead of printing it')
    s.add_argument('--detail', action='store_true', help='tasks: full plan with settings, warnings and manual facts per step')
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser('mcq', help='answer multiple-choice questions (typed, pasted or from a file)',
                       description='Three ways: (1) type it inline: cvassist mcq "Question? A) .. B) .. C) .. D) .."  '
                                   '(2) run "cvassist mcq" alone and paste one or many questions, then press Enter twice  '
                                   '(3) give a .txt file. Answers are printed with the fact they rely on. A "Correct '
                                   'Answer: X" line is hidden from the engine and only used to mark it right/wrong.')
    s.add_argument('text', nargs='*', help='the question with its options in quotes, or a .txt file; leave empty to paste')
    s.add_argument('--out', help='with a file: save a markdown answer sheet instead of printing')
    s.add_argument('--model', help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_mcq)

    s = sub.add_parser('plan', help='full step-by-step plan for a task',
                       description='Detailed plan: method, every step with reason, settings, code, warnings and manual '
                                   'pages, plus the full script.')
    s.add_argument('task', help='the task in quotes, or a .txt file containing it')
    s.add_argument('--out', help='save the plan (markdown) to this file')
    s.add_argument('--script', help='also save the runnable Python script to this file')
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser('search', help='look up topics in the knowledge base')
    s.add_argument('query')
    s.add_argument('--limit', type=int, default=5)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser('code', help='EXPERIMENTAL: new code for a custom question (needs the optional code model)',
                       description='Writes code for a question like: Write solve(img) that ... returns ... . The code is '
                                   'checked and repaired; if it still fails, the tested planner script is shown instead. '
                                   'About 55%% of custom tasks come out fully correct: always test the code.')
    s.add_argument('question')
    s.add_argument('--out', help='save the code to this file')
    s.add_argument('--model', help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_code)

    s = sub.add_parser('check', help='check that this machine has everything installed')
    s.set_defaults(fn=cmd_check)
    sub.add_parser('help', help='show help with examples')

    args = p.parse_args(argv)
    if args.command in (None, 'help'):
        print(HELP)
        return
    args.fn(args)


if __name__ == '__main__':
    main()
