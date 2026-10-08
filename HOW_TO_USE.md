# cvassist: how to use it

Offline OpenCV / computer-vision study assistant. No internet needed after setup, no large AI model.
It answers theory questions, plans and writes code for lab-style tasks, and answers MCQs, all from a
knowledge base built from the course lab manuals (Labs 01-06), with page references.

## First thing to run

Open a terminal **in this folder** and run:

```text
cvassist help
```

It lists every command with examples. For the options of one command: `cvassist ask --help`.

> In **PowerShell** type `.\cvassist` instead of `cvassist` (PowerShell does not run files from the current
> folder by name). In **cmd** plain `cvassist` works. Anywhere: `python cvassist.py help`.

## Setup on a new computer (once)

1. Install **Python 3.10, 3.11 or 3.12** (tick "Add Python to PATH" on Windows).
2. Copy this whole folder, **including `models/` and `outputs/`**.
3. In a terminal in this folder:

   ```text
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt        (Windows)
   .venv/bin/python -m pip install -r requirements.txt                (macOS / Linux)
   ```

4. Check everything: `cvassist check` (on macOS / Linux: `.venv/bin/python cvassist.py check`).
   Every line should say OK; it ends with "All set."

Needed in `models/`: `nli-deberta-v3-base` (MCQs, 0.24 GB) and `ms-marco-MiniLM-L-6-v2` (better answers, 23 MB).
OpenCV must be version 4.x (`check` warns if it is not).

## The commands

| Command | Use it for | Example |
|---|---|---|
| `ask` | Theory questions **and** tasks (main command) | `cvassist ask "Why do we blur before Canny?"` |
| `mcq` | Multiple-choice questions: typed, pasted or in a file | `cvassist mcq "Question? A) .. B) .. C) .. D) .."` |
| `plan` | Long, detailed plan for a task + script file | `cvassist plan "Count coins using Hough circles" --out plan.md --script solution.py` |
| `search` | Quick lookup in the knowledge base | `cvassist search "ratio test"` |
| `code` | EXPERIMENTAL new code for a custom question | `cvassist code "Write solve(img) that returns the number of coins"` |
| `check` | Is this machine set up correctly? | `cvassist check` |

Add `--out file.md` to `ask`, `mcq` or `plan` to save the answer to a file (open it in VS Code and press
Ctrl+Shift+V to read it formatted).

## Writing `ask` questions

**Theory question** (starts with what / why / how / explain / define / difference): you get the answer, an
explanation, key points with manual pages, common mistakes, and code for "how to" questions.

**Task** (detect / count / segment / implement / "You are working on ..."): you get the approach, numbered steps
with the reason for each, the complete runnable code, and its limitations. Numbers in the task become settings
(K=4, 25x25, 45 degrees, yellow, 12 screens). A long task can be put in a .txt file: `cvassist ask task.txt`.

Control the answer by adding words to the question:

| Add | You get |
|---|---|
| `in one line` | one sentence |
| `briefly` | the answer + 2 points |
| `in 3 points` (any number) | a numbered list |
| `in detail` | everything |
| `with code` | a runnable code example added |
| `only code` | just the code |

Examples:

```text
cvassist ask "What is the difference between affine and perspective transformation? in one line"
cvassist ask "Explain watershed segmentation in 3 points"
cvassist ask "How does Otsu thresholding work? with code"
cvassist ask "You are working on an autonomous vehicle project; detect lane markings using Hough lines" --out lanes.md
```

## MCQs: three ways

1. **Type one inline:** `cvassist mcq "Why is the image blurred before Canny? A) more contrast B) reduce noise C) grayscale D) thicker edges"`
2. **Paste many:** run `cvassist mcq` alone, paste the questions, press Enter on an empty line twice.
3. **From a file:** `cvassist mcq mcqs.txt` prints the answers; add `--out answers.md` for an answer-sheet file.

Each answer shows the chosen option and the knowledge-base fact it relied on (with manual page).

## MCQ format (pasting or file)

Plain text, a blank line between questions. "Correct Answer" lines are optional: if present they are hidden
from the engine and only used to mark the answer sheet right or wrong.

```text
1. Why is the image blurred before applying Canny?
A) To increase contrast
B) To reduce noise that would create false edges
C) To convert it to grayscale
D) To make edges thicker
Correct Answer: B
```

## How good is it (measured)

| Part | Result |
|---|---|
| `ask` theory, first sentence correct | 18/20 on new questions (20/20 after fixing the 2 gaps found) |
| `ask` / `plan` tasks, right method chosen | 115/115 task wordings; generated code checked on 20 synthetic tests |
| `mcq` | about 95% on the course question sets, about 60-70% on new hard conceptual questions |
| `code` (experimental) | about 55% fully correct on custom tasks: always test the code |

Answers come only from the knowledge base. A fact marked *CV fundamentals, not from the manual* is standard
textbook knowledge; everything else cites the lab manual page.

## Optional: the experimental `code` command

Needs `llama-cpp-python` and a code model in `models/qwen2.5-coder/` (the 1.5B model, 1.1 GB, gives about 55%;
the 0.5B model is too weak). Install: `.venv\Scripts\python.exe -m pip install llama-cpp-python==0.3.36
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`.
