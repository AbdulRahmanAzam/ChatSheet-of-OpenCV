"""Answer multiple-choice CV questions offline from the knowledge base (prototype).

Each option is scored by how well "question + option" matches the best passage in
the KB (curated topics, corrections and the manuals' page text), using TF-IDF
cosine similarity. The option with the highest score wins; the winning passage is
returned as evidence.

Usage:
    python work/mcq_answer.py work/eval/mcq_features_edges.txt   # evaluate a key file
"""
from pathlib import Path
import json, math, re, sys
from collections import Counter

import numpy as np

KB = Path(__file__).resolve().parents[1] / 'outputs' / 'cv_knowledge_base'
STOP = set('the an is are of to for and or in on it by be as at with from this that which what into its can each '
           'used use using only often typically usually also'.split())


def stem(token):
    for suffix in ('ations', 'ation', 'wards', 'ward', 'ings', 'ing', 'ies', 'ed', 'es', 's'):
        if token.endswith(suffix) and len(token) - len(suffix) >= (2 if 'ward' in suffix else 4):
            return token[:-len(suffix)] + ('y' if suffix == 'ies' else '')
    return token


def tokens(text):
    text = text.lower().replace('×', 'x').replace('→', ' to ')   # 2×3 -> 2x3, arrows -> "to"
    words = [stem(w) for w in re.findall(r'[a-z0-9_]+', text) if w not in STOP]
    # Adjacent word pairs make order count: "atan2 dy dx" differs from "atan2 dx dy".
    return words + [a + '__' + b for a, b in zip(words, words[1:])]


def load_passages():
    rows = lambda name: [json.loads(l) for l in (KB / 'records' / name).read_text(encoding='utf-8').splitlines() if l.strip()]
    passages = []
    for f in rows('facts.jsonl'):
        passages.append((f['id'] + ' ' + f['source_refs'][0], f['statement']))
    for t in rows('topics.jsonl'):
        text = ' '.join([t['title'], ' '.join(t['aliases']), t['answer'], t['when_to_use'], t['pitfalls'], t['code']])
        passages.append(('topic:' + t['id'], text))
    for c in rows('corrections.jsonl'):
        passages.append(('correction:' + c['id'], c['issue'] + ' ' + c['correction']))
    for p in rows('pages.jsonl'):
        if 'tasks' in p['source_id']:
            continue
        words = p['text'].split()
        for i in range(0, max(len(words), 1), 40):          # 80-word windows, 50% overlap
            chunk = ' '.join(words[i:i + 80])
            if chunk:
                passages.append((f"{p['id']}#w{i}", chunk))
    return passages


class Index:
    def __init__(self, passages):
        self.ids = [p[0] for p in passages]
        self.texts = [p[1] for p in passages]
        docs = [Counter(tokens(t)) for t in self.texts]
        df = Counter(w for d in docs for w in d)
        self.vocab = {w: i for i, w in enumerate(df)}
        n = len(docs)
        self.idf = np.array([math.log((n + 1) / (df[w] + 1)) + 1 for w in self.vocab])
        self.matrix = np.zeros((n, len(self.vocab)), np.float32)
        for r, d in enumerate(docs):
            for w, c in d.items():
                self.matrix[r, self.vocab[w]] = (1 + math.log(c)) * self.idf[self.vocab[w]]
        self.matrix /= np.linalg.norm(self.matrix, axis=1, keepdims=True) + 1e-9

    def vector(self, text):
        v = np.zeros(len(self.vocab), np.float32)
        for w, c in Counter(tokens(text)).items():
            if w in self.vocab:
                v[self.vocab[w]] = (1 + math.log(c)) * self.idf[self.vocab[w]]
        return v / (np.linalg.norm(v) + 1e-9)

    def best(self, text):
        sims = self.matrix @ self.vector(text)
        i = int(sims.argmax())
        return float(sims[i]), i


# "Which ... is NOT ..." / "... EXCEPT" ask for the unsupported option; "Why can it not ..." does not.
ALL_OF = re.compile(r'^\s*all of (the )?(above|these)\s*$', re.I)
NONE_OF = re.compile(r'^\s*none of (the )?(above|these)\s*$', re.I)


# Ask-for-the-false-option questions: "Which is NOT ...", "... is NOT mentioned", "all EXCEPT".
# "preserved by X but NOT necessarily by Y" asks for a true statement and must not trigger.
_NEG = re.compile(r"\b(is|are|was|were|does|do|would)\s+not\b|\bnot\s+(a|an|mentioned|used|shown|true|correct|part)\b|\bexcept\b", re.I)


def is_negated(question):
    q = re.sub(r"\bbut\s+not\b.*", "", question, flags=re.I)
    return bool(_NEG.search(q))


class _Neg:
    """Back-compat shim: NEG_CAPS.search / NEG_WHICH.search both route to is_negated."""
    @staticmethod
    def search(question):
        return is_negated(question)


NEG_CAPS = NEG_WHICH = _Neg()


def pick(question, options, scores):
    """Turn per-option evidence scores into a choice, handling NOT / all-of / none-of questions."""
    special = {j for j, o in enumerate(options) if ALL_OF.match(o) or NONE_OF.match(o)}
    regular = [j for j in range(len(options)) if j not in special]
    if is_negated(question):
        # "Which is NOT ...": the option with the weakest support in the KB.
        return min(regular, key=lambda j: scores[j])
    best = max(regular, key=lambda j: scores[j])
    for j in special:
        if ALL_OF.match(options[j]):
            # All regular options well supported -> "all of the above".
            if min(scores[i] for i in regular) >= 0.6 * scores[best]:
                return j
    return best


def answer(index, question, options, k=8):
    """Return (best option index, per-option scores, evidence passage id).

    Retrieve-then-read: only the k passages most relevant to the question (or to any
    question+option pair) are eligible evidence, so an option cannot win by matching an
    unrelated passage.
    """
    qsims = index.matrix @ index.vector(question)
    pool = set(np.argsort(-qsims)[:k])
    for opt in options:
        pool.update(np.argsort(-(index.matrix @ index.vector(question + ' ' + opt)))[:2])
    pool = sorted(pool, key=lambda i: -qsims[i])[:k + 2 * len(options)]
    sub = index.matrix[pool]
    scored = []
    for opt in options:
        # Joint match (question + option) plus the option's own words, weighted by passage relevance.
        sims = (sub @ index.vector(question + ' ' + opt) + sub @ index.vector(opt)) * (0.5 + qsims[pool])
        j = int(sims.argmax())
        s, i = float(sims[j]), pool[j]
        # Options that add no matching words of their own get no credit for the question's words alone.
        own = index.vector(opt)
        s *= 1.0 if own.any() else 0.5
        scored.append((s, i))
    scores = [s for s, _ in scored]
    k = pick(question, options, scores)
    return k, [round(s, 3) for s, _ in scored], index.ids[scored[k][1]]


def main(path):
    items = [l.split('|') for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]
    index = Index(load_passages())
    by_section, wrong = {}, []
    for n, (section, q, *opts, key) in enumerate(items, 1):
        k, scores, evidence = answer(index, q, opts)
        ok = 'ABCD'[k] == key.strip()
        tot = by_section.setdefault(section, [0, 0]); tot[0] += ok; tot[1] += 1
        if not ok:
            wrong.append(f"Q{n} [{section}] {q} -> picked {'ABCD'[k]} ({opts[k]}), key {key} ({opts['ABCD'.index(key.strip())]}) via {evidence}")
    correct = sum(v[0] for v in by_section.values())
    print(f'score {correct}/{len(items)} = {correct / len(items):.0%}')
    for s, (c, t) in by_section.items():
        print(f'  {s:11s} {c}/{t}')
    print('\nwrong:'); print('\n'.join(wrong))


if __name__ == '__main__':
    main(sys.argv[1])
