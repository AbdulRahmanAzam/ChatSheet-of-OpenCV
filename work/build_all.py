"""Test and rebuild the whole knowledge base in order. Stops at the first failure.

Usage (from the project root): .venv/Scripts/python.exe work/build_all.py
"""
from pathlib import Path
import subprocess, sys

HERE = Path(__file__).resolve().parent
KB = HERE.parent / 'outputs' / 'cv_knowledge_base'
STEPS = [
    HERE / 'test_supplement.py',            # supplemental snippets
    KB / 'solutions' / 'run_synthetic.py',  # 41 worked tasks, 56 checks
    HERE / 'build_apis.py',
    HERE / 'build_content.py',
    HERE / 'test_topics.py',                # every topic: setup + code
    HERE / 'build_index.py',
    HERE / 'eval_search.py',
    HERE / 'eval_mcq.py',                   # exam-style multiple choice
    HERE / 'plan_selftest.py',              # planner: generated code gives the right answer on synthetic truth
    HERE / 'plan_eval.py',                  # planner: right method for lab tasks + two frozen hold-outs
    HERE / 'build_plans.py',                # planner: plan + script for every lab task
    HERE / 'build_coverage.py',
]

for step in STEPS:
    print(f'== {step.name}', flush=True)
    if subprocess.run([sys.executable, str(step)], cwd=HERE.parent).returncode:
        sys.exit(f'FAILED at {step.name}')
print('== knowledge base rebuilt and verified')
