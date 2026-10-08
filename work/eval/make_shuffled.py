"""Write option-shuffled copies (*.shuf.txt) of MCQ key files, remapping the answer letter.

Questions whose options refer to positions ("All of the above", "Both A and B", "Neither A nor B")
keep their original order. Seeded, so the copies are reproducible.
"""
from pathlib import Path
import random, re, sys

POSITIONAL = re.compile(r'\b(all|none|both|neither) (of the above|of these|[a-d] (and|nor) [a-d])\b', re.I)


def shuffle_file(src, seed=7):
    rng = random.Random(seed)
    out = []
    for line in src.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            out.append(line)
            continue
        section, q, *opts, key = line.split('|')
        correct = opts['ABCD'.index(key.strip())]
        if not any(POSITIONAL.search(o) for o in opts):
            rng.shuffle(opts)
        out.append('|'.join([section, q, *opts, 'ABCD'[opts.index(correct)]]))
    dst = src.with_suffix('.shuf.txt')
    dst.write_text('\n'.join(out) + '\n', encoding='utf-8')
    return dst


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(shuffle_file(Path(p)))
