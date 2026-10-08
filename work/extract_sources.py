from pathlib import Path
from pypdf import PdfReader
import hashlib, json, shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'cv_knowledge_base'
names = ['Lab 01 Manual.pdf','Lab 01 Tasks.pdf','Lab 02 Tasks.pdf','Lab 03 Manual.pdf','Lab 03 Tasks.pdf','Lab 04 Manual.pdf','Lab 05 Manual.pdf','Lab 05 Tasks.pdf','Lab Manual 06.pdf','Lab 06 Tasks.pdf']
for sub in ['sources','extracted','records','solutions','reports']:
    (OUT/sub).mkdir(parents=True, exist_ok=True)
manifest=[]
pages=[]
for name in names:
    src=Path('C:/Users/Abdul Basit/Downloads')/name
    dest=OUT/'sources'/name
    shutil.copy2(src,dest)
    sid=name.removesuffix('.pdf').lower().replace(' ','_')
    reader=PdfReader(src)
    docpages=[]
    for i,p in enumerate(reader.pages,1):
        txt=p.extract_text() or ''
        row=dict(id=f'{sid}:p{i}',source_id=sid,page=i,text=txt,extraction='pypdf',character_count=len(txt),source_file='sources/'+name)
        pages.append(row); docpages.append(row)
    (OUT/'extracted'/f'{sid}.txt').write_text('\n\n'.join(f'=== PAGE {p["page"]} ===\n{p["text"]}' for p in docpages),encoding='utf-8')
    item=dict(id=sid,file='sources/'+name,title=name,pages=len(reader.pages),bytes=src.stat().st_size,sha256=hashlib.sha256(src.read_bytes()).hexdigest(),page_characters=[p['character_count'] for p in docpages])
    manifest.append(item)
    print(json.dumps(item))
(OUT/'records'/'sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'records'/'pages.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pages),encoding='utf-8')
