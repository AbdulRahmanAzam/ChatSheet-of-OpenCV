from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'/'cv_knowledge_base'
sources=json.loads((OUT/'records'/'sources.json').read_text())
pages=[]
for source in sources:
    chunks=[]
    for n in range(1,source['pages']+1):
        pid=f'{source["id"]}_p{n:03}'
        raw=json.loads((OUT/'extracted'/'ocr'/f'{pid}.json').read_text(encoding='utf-8-sig'))
        txt='\n'.join(line['text'] for line in raw['lines'])
        page=dict(id=f'{source["id"]}:p{n}',source_id=source['id'],page=n,text=txt,extraction='Windows.Media.Ocr',review_status='raw_ocr_unverified',image=f'extracted/page_images/{pid}.jpg',source_file=source['file'])
        pages.append(page);chunks.append(f'=== PAGE {n} ===\n{txt}')
    (OUT/'extracted'/f'{source["id"]}.txt').write_text('\n\n'.join(chunks),encoding='utf-8')
(OUT/'records'/'pages.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pages),encoding='utf-8')
print('Merged',len(pages),'pages;',sum(len(p['text']) for p in pages),'characters')
