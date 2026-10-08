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

### Download the MCQ model (required for `mcq`)

`nli-deberta-v3-base` is too big for git (233 MB). Download it from the
[Releases page](https://github.com/AbdulRahmanAzam/ChatSheet-of-OpenCV/releases/latest) and place it at
`models/nli-deberta-v3-base/model_quint8_avx2.onnx`:

```bash
gh release download models-v1 -R AbdulRahmanAzam/ChatSheet-of-OpenCV -p model_quint8_avx2.onnx -D models/nli-deberta-v3-base
```

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
