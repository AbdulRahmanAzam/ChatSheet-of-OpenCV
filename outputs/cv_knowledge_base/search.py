"""Read-only offline knowledge lookup using Python's standard library.
Usage: python search.py "why Gaussian before Canny" --limit 5
Raw OCR and out-of-domain course appendices are opt-in; this is retrieval, not generation.
"""
from pathlib import Path
import argparse,json,re,sqlite3

ROOT=Path(__file__).resolve().parent
STOP={'the','a','an','is','are','of','to','for','and','or','in','on','it','can','i','you','me','my','how','what','why','do','does','with','use','using','please','tell','about','this','that','image','particular'}


def stem(token):
    """Strip common English suffixes; keeps at least 4 characters so short API names survive."""
    for suffix in ('ations','ation','ings','ing','ies','ed','es','s'):
        if token.endswith(suffix) and len(token)-len(suffix)>=4:
            return token[:-len(suffix)]+('y' if suffix=='ies' else '')
    return token


def search(query,limit=5,include_raw=False,include_appendix=False):
    if not query.strip(): return []
    query=re.sub(r'\bscar\b','scharr',query,flags=re.I)
    query=re.sub(r'BruteForceMatcher','BFMatcher',query,flags=re.I)
    tokens=[t.lower() for t in re.findall(r'[\w]+',query) if t.lower() not in STOP]
    if not tokens: return []
    # Quote every token. User input never becomes SQL or an FTS operator.
    # A light suffix stem plus FTS prefix match lets "blurring" find "blur", "objects" find "object".
    phrase=' OR '.join('"'+t.replace('"','""')+'"*' for t in dict.fromkeys(stem(t) for t in tokens))
    conn=sqlite3.connect((ROOT/'knowledge.sqlite').as_uri()+'?mode=ro',uri=True)
    conn.row_factory=sqlite3.Row
    try:
        rows=conn.execute('''SELECT r.*, bm25(search_index,0,0,6,4,1) AS score
            FROM search_index JOIN records r ON r.id=search_index.id
            WHERE search_index MATCH ?
              AND (? OR r.kind != 'raw_page')
              AND (? OR r.scope = 'core')
            ORDER BY score LIMIT ?''',(phrase,int(include_raw),int(include_appendix),max(1,min(int(limit),100)))).fetchall()
        return [dict(id=r['id'],kind=r['kind'],title=r['title'],record=json.loads(r['payload'])) for r in rows]
    finally: conn.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query'); parser.add_argument('--limit',type=int,default=5)
    parser.add_argument('--raw',action='store_true',help='Include unverified OCR; consult scans before relying on code/equations')
    parser.add_argument('--appendix',action='store_true',help='Include supporting Python and sensor exercises')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args(); results=search(args.query,args.limit,args.raw,args.appendix)
    if args.json: print(json.dumps(results,ensure_ascii=False,indent=2)); return
    if not results: print('No matching record. This corpus cannot answer every CV question.'); return
    for item in results:
        print(f"\n{item['id']} | {item['kind']} | {item['title']}")
        r=item['record']
        for field in ['statement','note','answer','when_to_use','pitfalls','correction','assumptions_and_limits','text']:
            if field in r: print(f"{field}: {r[field]}")
        if 'solution_steps' in r:
            for n,step in enumerate(r['solution_steps'],1): print(f'{n}. {step}')
        if 'code' in r: print(r['code'])
        if 'entry_point' in r: print('Implementation:',r['entry_point'])
        if 'source_refs' in r: print('Sources:',', '.join(r['source_refs']))

if __name__=='__main__': main()
