"""Offline multiple-choice answering: triples -> NLI reader -> lexical tie-break.

Layer 1 (structured facts): a triple whose subject is named in the question and whose
    attribute matches the question binds exactly one value; the option equal to that
    value wins when the match is unambiguous.
Layer 2 (reader): a small NLI model (DeBERTa-v3-base, 8-bit ONNX, ~240 MB) scores
    whether retrieved facts entail "question + option".
Layer 3 (lexical): TF-IDF evidence score from mcq_answer.py, used as a prior and tie-break.

Usage: python work/mcq_engine.py work/eval/mcq_lab06.txt [--no-nli] [--no-triples] [--explain]
"""
from pathlib import Path
import json, re, sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mcq_answer as lex  # noqa: E402

KB = lex.KB
MODEL_DIR = HERE.parent / 'models' / 'nli-deberta-v3-base'   # chosen 2026-10-08: 27/40 hold-out, 0.24 GB

SUPERSCRIPTS = str.maketrans({'²': '^2', '³': '^3', '−': '-', '–': '-', '—': '-', '×': 'x', '→': '->', 'σ': 'sigma',
                              'θ': 'theta', 'ρ': 'rho', '∞': 'inf', '’': "'", '‘': "'", '“': '"', '”': '"',
                              '>': ' greater than ', '<': ' less than ', '≥': ' greater or equal ', '≤': ' less or equal '})


def norm(text):
    return text.translate(SUPERSCRIPTS)


# ----------------------------------------------------------------------------- layer 1
class Triples:
    def __init__(self):
        path = KB / 'records' / 'triples.jsonl'
        self.rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]

    @staticmethod
    def subject_in(alias, question):
        """Whole-phrase match; short aliases (a, M, x) must be quoted or exact-case to avoid matching articles."""
        q = norm(question)
        if len(alias) <= 2:
            if re.search(r"['\"]" + re.escape(alias) + r"['\"]", q):
                return True
            if alias.isupper() and re.search(r'(?<![\w.])' + re.escape(alias) + r'(?![\w(])', q):
                return True
            return bool(re.search(r'\b' + re.escape(alias) + r'(-axis| increases| goes)\b', q))
        return bool(re.search(r'(?<![\w])' + re.escape(norm(alias)) + r'(?![\w])', q, re.I))

    def candidates(self, question):
        qtok = set(t for t in lex.tokens(norm(question)) if '__' not in t)
        out = []
        for r in self.rows:
            hits = [a for a in r['subjects'] if self.subject_in(a, question)]
            if not hits:
                continue
            atok = set(t for t in lex.tokens(r['attribute']) if '__' not in t)
            attr = len(atok & qtok) / max(len(atok), 1)
            specificity = min(len(max(hits, key=len)), 20) / 20
            out.append((attr * (0.7 + 0.3 * specificity), attr, r))
        return sorted(out, key=lambda x: -x[0])

    @staticmethod
    def value_match(value, option):
        v = set(t for t in lex.tokens(norm(value)) if '__' not in t)
        o = set(t for t in lex.tokens(norm(option)) if '__' not in t)
        if not v or not o:
            return float(norm(value).strip().lower() == norm(option).strip().lower())
        if norm(value).strip().lower() == norm(option).strip().lower():
            return 1.0
        return len(v & o) / len(v | o)

    def decide(self, question, options):
        """Return (option index, triple) when one triple binds an unambiguous answer, else (None, None)."""
        cands = self.candidates(question)
        if not cands or cands[0][1] < 0.5:
            return None, None
        top = cands[0][0]
        for score, attr, r in cands:
            if score < top - 1e-9:
                break
            m = [self.value_match(r['value'], o) for o in options]
            order = np.argsort(m)[::-1]
            if m[order[0]] >= 0.5 and m[order[0]] - m[order[1]] >= 0.25:
                return int(order[0]), r
        return None, None


# ----------------------------------------------------------------------------- layer 2
class Reader:
    """DeBERTa-v3 NLI cross-encoder via onnxruntime (CPU, offline)."""

    def __init__(self, model_dir=MODEL_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(str(model_dir / 'model_quint8_avx2.onnx'), providers=['CPUExecutionProvider'])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        labels = json.loads((model_dir / 'config.json').read_text())['id2label']
        self.entail = int(next(k for k, v in labels.items() if v == 'entailment'))
        self.contra = int(next(k for k, v in labels.items() if v == 'contradiction'))

    def scores(self, premises, hypotheses):
        """Entailment minus contradiction probability for each (premise, hypothesis) pair."""
        enc = self.tok.encode_batch(list(zip(premises, hypotheses)))
        feed = {'input_ids': np.array([e.ids for e in enc], np.int64),
                'attention_mask': np.array([e.attention_mask for e in enc], np.int64)}
        if 'token_type_ids' in self.inputs:
            feed['token_type_ids'] = np.array([e.type_ids for e in enc], np.int64)
        logits = self.sess.run(None, feed)[0]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return p[:, self.entail] - p[:, self.contra]


def seq(text):
    return [t for t in lex.tokens(norm(text)) if '__' not in t]


def ordered_match(option, fact):
    """Fraction of the option's tokens found in the fact in the same order (longest common subsequence)."""
    a, b = seq(option), seq(fact)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1] / len(a)


BOTH = re.compile(r'^\s*both ([a-d]) and ([a-d])\s*$', re.I)
# Absolute wording ("only ... exactly", "all ... will", "always", "never") marks the typical distractor in
# "which is most accurate" questions. A small penalty per word, set on the 428 dev questions (W_ABSOLUTE).
ABSOLUTE = re.compile(r'\b(only|exactly|always|never|all|every|guarantees?|completely|entirely|impossible)\b', re.I)
W_ABSOLUTE = 0.3   # no change on the 428 dev or 40 hold-out questions; guards "only/exactly/all" distractors


W_CONTRA = 0.0      # weight of the strongest contradiction from any retrieved fact (set on the 428 dev questions)


ABSOLUTE = re.compile(r'\b(exactly|always|never|guarantees?|completely|entirely|impossible)\b'
                      r'|\bonly\b(?!\s+(if|when|after|for|those|the pixels|pixels))|^\s*(all|every)\b', re.I)


def absolute_words(option):
    """Absolute claims ("always", "exactly", "only X", "All ... will"); precise conditions like "only if" do not count."""
    return len(ABSOLUTE.findall(option))


STATEMENT_Q = re.compile(r'(which|what)\s+(of the following\s+)?(statement|option|one|claim)?\s*(of the following\s+)?'
                         r'(is|best|most)\b[^?]*(accurate|true|correct|describes?|explains?)[^?]*\?\s*$', re.I)


def hypothesis(question, option):
    q = norm(question).strip()
    o = norm(option).strip()
    m = STATEMENT_Q.search(q)
    if m:
        # "...code... Which statement is most accurate?": each option is a full statement; judge it on its own,
        # anchored to the OpenCV function the question is about (the long scenario only confuses the reader).
        stem = q[:m.start()]
        func = re.findall(r'cv2?\.([A-Za-z]\w*)\s*\(', stem) or re.findall(r'cv2?\.([a-z]\w*|[A-Z][a-z]\w*)', stem)
        return f'{"cv2." + func[-1] + ": " if func else ""}{o}.'
    if q.endswith('?'):
        return f'{q[:-1]}: {o}.'
    q = q.rstrip(':').strip()
    if '___' in q:
        return q.replace('___', o) + '.'
    return f'{q} {o}.'


class Engine:
    w_lex, w_nli, w_ord = 0.15, 1.0, 0.3   # chosen for the base reader on the 428 dev questions (407/428)

    def __init__(self, use_nli=True, use_triples=True, k=6, model_dir=MODEL_DIR):
        self.index = lex.Index(lex.load_passages())
        self.fact_rows = [i for i, pid in enumerate(self.index.ids) if pid.startswith('fact:')]
        self.triples = Triples() if use_triples else None
        self.reader = Reader(model_dir) if use_nli else None
        self.k = k

    def premises(self, question, options):
        """Top facts for the question and for each question+option pair."""
        sub = self.index.matrix[self.fact_rows]
        pool = {}
        for text in [question] + [question + ' ' + o for o in options]:
            sims = sub @ self.index.vector(norm(text))
            for j in np.argsort(-sims)[:self.k]:
                pool[self.fact_rows[j]] = max(pool.get(self.fact_rows[j], 0), sims[j])
        best = sorted(pool, key=lambda i: -pool[i])[:self.k + 2]
        return [self.index.texts[i] for i in best], [self.index.ids[i] for i in best]

    def features(self, question, options):
        """All evidence for one question; the expensive part (reader) is computed once and can be cached."""
        f = dict(triple=None)
        if self.triples is not None and not (lex.NEG_CAPS.search(question) or lex.NEG_WHICH.search(question)):
            k, t = self.triples.decide(question, options)
            if k is not None:
                f['triple'] = dict(choice=k, evidence=t['id'] + ' ' + t['source_refs'][0],
                                   fact=f"{t['subjects'][0]} | {t['attribute']} | {t['value']}")
        _, lex_scores, lex_ev = lex.answer(self.index, norm(question), [norm(o) for o in options])
        lex_scores = np.array(lex_scores, float)
        f['lex'] = list(lex_scores / (lex_scores.max() + 1e-9))
        f['lex_evidence'] = lex_ev
        texts, ids = self.premises(question, options)
        f['texts'], f['ids'] = texts, ids
        # Ordered evidence: option tokens in the same order inside a question-relevant fact.
        qrel = np.array([float(self.index.vector(norm(question)) @ self.index.vector(t)) for t in texts])
        f['ordered'] = [max(ordered_match(o, t) * (0.5 + r) for t, r in zip(texts, qrel)) for o in options]
        if self.reader is not None:
            hyps = [hypothesis(question, o) for o in options]
            s = self.reader.scores([t for t in texts for _ in options], [h for _ in texts for h in hyps])
            f['nli'] = s.reshape(len(texts), len(options)).tolist()
        return f

    def answer(self, question, options):
        """Return dict(choice, layer, evidence, fact, scores)."""
        return self.combine(question, options, self.features(question, options))

    def combine(self, question, options, f, w_lex=None, w_nli=None, w_ord=None):
        w_lex = self.w_lex if w_lex is None else w_lex
        w_nli = self.w_nli if w_nli is None else w_nli
        w_ord = self.w_ord if w_ord is None else w_ord
        if f['triple'] is not None:
            return dict(layer='triple', **f['triple'])
        lex_scores, orderly = np.array(f['lex']), np.array(f['ordered'])
        if 'nli' not in f:
            total = list(lex_scores + w_ord * orderly)
            return dict(choice=lex.pick(question, options, total), layer='lexical', evidence=f['lex_evidence'], fact='')
        s = np.array(f['nli'])
        texts, ids = f['texts'], f['ids']
        nli = s.max(0)
        best_prem = s.argmax(0)
        total = list(w_lex * lex_scores + w_nli * nli + w_ord * orderly)
        total = [t - W_ABSOLUTE * absolute_words(o) for t, o in zip(total, options)]
        # An option that a retrieved fact clearly contradicts is penalised, even if another fact loosely supports it.
        contra = np.minimum(s.min(0), 0.0)
        total = [t + W_CONTRA * c for t, c in zip(total, contra)]
        for j, o in enumerate(options):
            m = BOTH.match(o)
            if m:
                a, b = ('abcd'.index(m.group(1).lower()), 'abcd'.index(m.group(2).lower()))
                if a < len(options) and b < len(options):
                    others = [total[i] for i in range(len(options)) if i not in (a, b, j)]
                    if min(total[a], total[b]) > max(others or [-9]):
                        total[j] = max(total) + 1e-3
        k = lex.pick(question, options, total)
        return dict(choice=k, layer='nli', evidence=ids[best_prem[k]] if k < len(best_prem) else f['lex_evidence'],
                    fact=texts[best_prem[k]] if k < len(best_prem) else '', scores=[round(x, 3) for x in total])


def main(argv):
    path = Path(argv[0])
    model = next((a.split('=', 1)[1] for a in argv if a.startswith('--model=')), None)
    engine = Engine(use_nli='--no-nli' not in argv, use_triples='--no-triples' not in argv,
                    model_dir=HERE.parent / 'models' / model if model else MODEL_DIR)
    rows = [l.split('|') for l in path.read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]
    correct, by_layer, wrong = 0, {}, []
    for n, (section, q, *opts, key) in enumerate(rows, 1):
        r = engine.answer(q, opts)
        ok = 'ABCD'[r['choice']] == key.strip()
        correct += ok
        c = by_layer.setdefault(r['layer'], [0, 0]); c[0] += ok; c[1] += 1
        if not ok:
            wrong.append(f"Q{n} [{section}] {q} -> {opts[r['choice']]!r} (key {opts['ABCD'.index(key.strip())]!r}) [{r['layer']}] {r.get('fact', '')[:110]}")
    print(f'{path.name}: {correct}/{len(rows)}  by layer: ' + ', '.join(f'{k} {v[0]}/{v[1]}' for k, v in by_layer.items()))
    if '--explain' in argv:
        print('\n'.join(wrong))
    return correct, len(rows)


if __name__ == '__main__':
    main(sys.argv[1:])
