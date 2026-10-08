# ChatSheet of OpenCV (cvassist)

Offline OpenCV / computer-vision study assistant. Answers theory questions, plans and writes code for
lab-style tasks, and answers MCQs from a knowledge base built from lab manuals (Labs 01-06), with page refs.

## Install

Requires Python 3.10-3.12.

```bash
git clone https://github.com/AbdulRahmanAzam/ChatSheet-of-OpenCV.git
cd ChatSheet-of-OpenCV
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt      # Windows
.venv/bin/python -m pip install -r requirements.txt              # macOS / Linux
```

The required models are included (the 233 MB MCQ model is stored with Git LFS). Easiest: on GitHub click
**Code → Download ZIP** and extract. If you `git clone`, install [Git LFS](https://git-lfs.com) first
(`git lfs install`) so the model file downloads instead of a small pointer file.

Optional (experimental `code` command): download `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` from
[Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF) into
`models/qwen2.5-coder/`, plus `pip install llama-cpp-python`.

### Verify

```bash
python cvassist.py check
```

## Use

```bash
python cvassist.py help
python cvassist.py ask "Why do we blur before Canny?"
python cvassist.py mcq "Question? A) .. B) .. C) .. D) .."
python cvassist.py plan "Count coins using Hough circles" --out plan.md --script solution.py
python cvassist.py search "ratio test"
```

On Windows `cvassist <command>` (cmd) or `.\cvassist <command>` (PowerShell) also works.

Full guide: [HOW_TO_USE.md](HOW_TO_USE.md).
