from pathlib import Path
import json,ast,sqlite3,html,re,hashlib
ROOT=Path(__file__).resolve().parents[1]/'outputs'/'cv_knowledge_base'
def read(name): return [json.loads(l) for l in (ROOT/'records'/name).read_text(encoding='utf-8').splitlines() if l.strip()]
def write(name,records): (ROOT/'records'/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
topics,tasks,corrections,pages=map(read,['topics.jsonl','tasks.jsonl','corrections.jsonl','pages.jsonl'])
sources=json.loads((ROOT/'records'/'sources.json').read_text(encoding='utf-8'))
by_page={p['id']:p for p in pages}
nodes={}
for path in sorted((ROOT/'solutions').glob('*.py')):
    content=path.read_text(encoding='utf-8')
    for node in ast.parse(content).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            nodes[path.stem+'.'+node.name]=(ast.get_source_segment(content,node),path.name,node.lineno)
snippets=[]
for t in tasks:
    code,file,line=nodes[t['entry_point']]
    snippets.append(dict(id='code:'+t['id'],title=t['title'],code=code,
        file='solutions/'+file,line=line,source_refs=t['source_refs'],scope=t['scope'],
        entry_point=t['entry_point'],dependencies=['cv_core.py','numpy','opencv-python-headless'],
        status='corrected_original_implementation',context='Import the module to obtain its helper functions/imports; this function/class body is not a standalone script.'))
groups=[]
def group(id,title,source,pnums,entry,selector='',notes=''):
    refs=[source+':p'+str(p) for p in pnums]
    code,file,line=nodes[entry]
    groups.append(dict(id=id,title=title,source_refs=refs,
        source_pages=[{'id':ref,'image':by_page[ref]['image'],'raw_ocr':by_page[ref]['text']} for ref in refs],
        entry_point=entry,output_selector=selector,corrected_code=code,
        file='solutions/'+file,line=line,notes=notes,
        extraction_status='Full original code preserved visually in the page scans; raw OCR is unverified. Runnable equivalent is a corrected reconstruction, not a verbatim transcription.',scope='core'))
for p,title,key in [(14,'Read/display color','color'),(15,'Grayscale display','gray'),(16,'Resize half size','half_size'),(17,'Gaussian blur31','gaussian31'),(18,'Left half crop','left_crop'),(19,'Text overlay','text'),(21,'Threshold100/max200','threshold100_max200'),(22,'Rotation45 same canvas','rotate_same_canvas'),(23,'Saturating addition','saturating_add'),(24,'Histogram equalization','equalized')]:
    group(f'M01-{p}',title,'lab_01_manual',[p],'manual_examples.basic_examples',key)
group('M01-20','Pandas RGB pixel summary','lab_01_manual',[20],'lab01.task09_optional_pandas',notes='Pandas is optional and outside the core dependency scope. NumPy equivalent is task09_rgb_statistics.')
for id,title,pnums,entry,key,note in [
 ('HOG','HOG extraction and visualization',[7],'manual_examples.feature_examples','hog','OpenCV HOG descriptor replacement; original skimage HOG visualization not reproduced bit-for-bit.'),
 ('LBP','Uniform local binary pattern',[8,9],'manual_examples.feature_examples','lbp_histogram','Circular8-neighbor replacement; border/sampling differs from scikit-image.'),
 ('HED','Histogram of edge directions',[10,11],'manual_examples.feature_examples','hed','Fix angle wrapping and apply edge mask.'),
 ('HIG','Histogram of intensity gradients',[12,13],'manual_examples.feature_examples','hig','Fix angle wrapping; explicitly magnitude-weighted.'),
 ('TEXTURE','Texture energy and contrast',[14,15],'manual_examples.feature_examples','energy;contrast','Float intermediates; implement stated standard-deviation contrast.'),
 ('BOX','Box smoothing',[16],'manual_examples.filtering_examples','box',''),
 ('GAUSS','Gaussian kernel filtering',[17],'manual_examples.filtering_examples','gaussian2d;gaussian_separable','Correct1D column to2D filtering.'),
 ('DERIV','Sobel and Scharr',[17],'manual_examples.filtering_examples','sobel_x;sobel_y;scharr_x;scharr_y',''),
 ('EMBOSS','Emboss kernel',[17],'manual_examples.filtering_examples','emboss_signed','Keep signed response for analysis.'),
 ('STANDARD','Standard filtering',[17,18],'manual_examples.filtering_examples','correlation_same;convolution_full','Distinguish correlation from full convolution.'),
 ('VALID','Valid convolution',[18],'manual_examples.filtering_examples','convolution_valid','Crop valid support; borderconstant alone is insufficient.'),
 ('SAME','Same zero-padded convolution',[18,19],'manual_examples.filtering_examples','convolution_same_zero','Use zero padding, not reflected borders.'),
 ('STRIDE','Strided valid convolution',[19],'manual_examples.filtering_examples','convolution_stride2','Replace unsupported strides keyword with explicit subsampling.'),
 ('GRADIENT','Gradient magnitude threshold',[21],'cv_core.gradients','magnitude','Apply (magnitude>100).astype(np.uint8)*255 for original threshold demonstration.'),
 ('CANNY','Canny edge example',[23],'manual_examples.segmentation_examples','canny','Explicit Gaussian preprocessing documented.')]:
    group('M04-'+id,title,'lab_04_manual',pnums,entry,key,note)
for id,title,pnums,key in [('GLOBAL','Global threshold',[8],'global'),('ADAPT','Adaptive threshold',[8,9],'adaptive'),('OTSU','Otsu threshold',[9],'otsu'),('HSV','HSV mask',[9],'hsv_mask'),('CANNY','Canny edges',[10],'canny'),('REGION','Region growing',[10,11,12],'region_grow'),('WATER','Watershed markers',[13],'watershed'),('KMEANS','K-means colors',[15],'kmeans')]:
    group('M05-'+id,title,'lab_05_manual',pnums,'manual_examples.segmentation_examples',key)
for id,title,pnums,key in [('SOBEL','Sobel direction and magnitude',[7,8,9,10],'sobel_x;sobel_y;magnitude'),('CANNY','Gaussian and Canny',[12,13],'canny'),('LOG','Laplacian after Gaussian',[15,16],'laplacian_signed;laplacian_absolute_display;log_zero_crossings'),('SIFT','SIFT keypoint drawing',[22,23],'sift_keypoints_image')]:
    group('M06-'+id,title,'lab_manual_06',pnums,'manual_examples.edge_feature_examples',key)
write('snippets.jsonl',snippets);write('manual_examples.jsonl',groups)

qa=[]
for t in topics:
    qa.append(dict(id='qa:'+t['id']+':why',question='What is '+t['title']+' and when should I use it?',
        answer=t['answer']+' '+t['when_to_use'],topic_id=t['id'],source_refs=t['source_refs'],scope='core',
        status='authored retrieval examples; not independent evaluation data'))
    qa.append(dict(id='qa:'+t['id']+':pitfalls',question='What mistakes should I avoid with '+t['title']+'?',
        answer=t['pitfalls'],topic_id=t['id'],source_refs=t['source_refs'],scope='core',
        status='authored retrieval examples; not independent evaluation data'))
write('qa.jsonl',qa)

# Content provenance and scope are explicit; unverified OCR is opt-in at lookup.
records=[]
def add(id,kind,title,aliases,body,payload,scope='core'):
    records.append(dict(id=id,kind=kind,title=title,aliases=aliases,body=body,payload=payload,scope=scope))
for t in topics: add('topic:'+t['id'],'topic',t['title'],' '.join(t['aliases']),json.dumps(t,ensure_ascii=False),t)
for t in tasks: add(t['id'],'task',t['title'],t['entry_point'],json.dumps(t,ensure_ascii=False),t,t['scope'])
for c in corrections: add(c['id'],'correction',c['issue'],'',c['correction'],c)
for s in snippets: add(s['id'],'solution_code',s['title'],s['entry_point'],s['code'],s,s['scope'])
for g in groups:
    payload={k:v for k,v in g.items() if k!='source_pages'}
    add(g['id'],'manual_example',g['title'],g['entry_point'],g['notes']+' '+g['corrected_code'],payload)
api_path=ROOT/'records'/'apis.jsonl'
if api_path.exists():
    for api in read('apis.jsonl'): add(api['id'],'api',api['title'],' '.join(api['aliases']),json.dumps(api),api)
if (ROOT/'records'/'facts.jsonl').exists():
    for f in read('facts.jsonl'): add(f['id'],'fact',f['statement'],f['topic_id'],f['statement']+' '+f['note'],f)
for p in pages: add(p['id'],'raw_page',p['source_id']+' page '+str(p['page']),'',p['text'],p)
db=ROOT/'knowledge.sqlite'
conn=sqlite3.connect(db)
conn.executescript('''DROP TABLE IF EXISTS search_index; DROP TABLE IF EXISTS records; DROP TABLE IF EXISTS sources;
CREATE TABLE records(id TEXT PRIMARY KEY,kind TEXT NOT NULL,title TEXT,aliases TEXT,body TEXT,payload TEXT,scope TEXT);
CREATE TABLE sources(id TEXT PRIMARY KEY,title TEXT,payload TEXT);
CREATE VIRTUAL TABLE search_index USING fts5(id UNINDEXED,kind UNINDEXED,title,aliases,body,tokenize='unicode61');''')
conn.executemany('INSERT INTO records VALUES(?,?,?,?,?,?,?)',[(r['id'],r['kind'],r['title'],r['aliases'],r['body'],json.dumps(r['payload'],ensure_ascii=False),r['scope']) for r in records])
conn.executemany('INSERT INTO search_index VALUES(?,?,?,?,?)',[(r['id'],r['kind'],r['title'],r['aliases'],r['body']) for r in records])
conn.executemany('INSERT INTO sources VALUES(?,?,?)',[(s['id'],s['title'],json.dumps(s)) for s in sources])
conn.commit(); check=conn.execute('PRAGMA integrity_check').fetchone()[0]; conn.close()
assert check=='ok'

# Standalone reference browser: embedded data; no CDN, fetch(), web API or server.
browser_records=[]
for r in records:
    if r['kind']=='raw_page': continue
    p=r['payload']
    text=[]
    for k in ['answer','when_to_use','pitfalls','issue','correction','assumptions_and_limits','notes','signature','input_notes','output_notes']:
        if p.get(k): text.append(k.replace('_',' ').title()+': '+str(p[k]))
    if p.get('solution_steps'): text += [f'{i}. {s}' for i,s in enumerate(p['solution_steps'],1)]
    code=p.get('code',p.get('corrected_code',''))
    refs=[]
    for ref in p.get('source_refs',[]):
        if ref in by_page: refs.append({'label':ref,'href':by_page[ref]['image']})
    browser_records.append(dict(id=r['id'],kind=r['kind'],title=r['title'],aliases=r['aliases'],text='\n\n'.join(text),code=code,
                                refs=refs,scope=r['scope'],file=p.get('implementation',p.get('file',''))))
source_links=''.join(f'<li><a href="{html.escape(s["file"])}">{html.escape(s["title"])}</a> · {s["pages"]} pages · <a href="extracted/{s["id"]}.txt">raw OCR text</a></li>' for s in sources)
data=json.dumps(browser_records,ensure_ascii=False).replace('</','<\\/')
template='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Offline Computer Vision Knowledge Base</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#172839;background:#f5f7fa;margin:0}header,main{max-width:1120px;margin:auto;padding:28px}header{background:#122e43;color:white;max-width:none}header>div{max-width:1120px;margin:auto}h1{line-height:1.2;margin:10px 0}h2{font-size:21px}a{color:#096b9c}header a{color:#aee9ff}.tag{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#53718c}input,select{font:inherit;padding:12px;border:1px solid #a9b9c6;border-radius:6px}input[type=search]{width:min(700px,95%)}.controls{position:sticky;top:0;background:#f5f7fa;padding:14px 0;z-index:1;display:flex;gap:12px;flex-wrap:wrap}.card{background:white;border:1px solid #d6e0e6;border-radius:10px;padding:24px;margin:18px 0}.body{white-space:pre-wrap}pre{overflow:auto;background:#edf3f7;padding:16px;font-size:13px;line-height:1.45}.small{font-size:14px;color:#4e6577}.stats{display:flex;gap:24px;flex-wrap:wrap;margin:20px 0}details{margin:12px 0}button{font:inherit;padding:9px;border:1px solid #789;border-radius:5px;background:white;cursor:pointer}.ref{margin-right:12px}footer{margin:30px 0;color:#4e6577}</style>
<header><div><p>LOCAL REFERENCE LIBRARY · VERSION 0.1</p><h1>Computer vision, explained and worked through</h1>
<p>Manuals, corrected examples and exercise solutions. Search works entirely offline.</p>
<div class="stats"><span><b>10</b> source PDFs</span><span><b>135</b> preserved pages</span><span><b>61</b> topic records</span><span><b>41</b> worked exercises</span></div>
<a href="README.md">Start here</a> · <a href="HANDBOOK.md">Handbook</a> · <a href="TASK_SOLUTIONS.md">Task guide</a> · <a href="reports/COVERAGE.md">Coverage and limits</a></div></header>
<main><details><summary>Original manuals and extracted text</summary><p>OCR contains recognition errors. Use page scans for exact original code and equations. The corrected examples are separate reconstructions.</p><ul>__SOURCES__</ul></details>
<div class="controls"><input id="q" type="search" placeholder="Try: Gaussian blur, watershed, SIFT, screen detection" aria-label="Search knowledge base"><select id="kind" aria-label="Record type"><option value="">All curated records</option value="topic">Topics</option><option value="task">Tasks</option><option value="api">API reference</option><option value="manual_example">Manual examples</option><option value="solution_code">Solution code</option><option value="correction">Corrections</option></select><label><input id="appendix" type="checkbox"> Include course appendices</label></div>
<p id="count" class="small"></p><div id="results"></div><footer>Source-linked retrieval, not an LLM or trained predictor. No real-data accuracy is claimed. The supplied scans and external material retain their owners’ rights.</footer></main>
<script>const records=__DATA__;const q=document.getElementById('q'),kind=document.getElementById('kind'),appendix=document.getElementById('appendix'),results=document.getElementById('results');
function el(tag,text,cls){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x;}
function render(){const terms=q.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);const filtered=records.filter(r=>(appendix.checked||r.scope==='core')&&(!kind.value||r.kind===kind.value)).map(r=>{const title=(r.title+' '+r.aliases).toLowerCase(),body=(r.text+' '+r.code+' '+r.id).toLowerCase();return {r,score:terms.reduce((s,t)=>s+(title.includes(t)?5:body.includes(t)?1:0),0)};}).filter(x=>!terms.length||x.score>0).sort((a,b)=>b.score-a.score);results.replaceChildren();document.getElementById('count').textContent=filtered.length+' matching records · showing first 60';
for(const {r} of filtered.slice(0,60)){const c=el('article',undefined,'card');c.append(el('div',r.kind.replaceAll('_',' ')+' · '+r.id,'tag'),el('h2',r.title),el('div',r.text,'body'));
if(r.code){const d=el('details');d.append(el('summary','View code'),el('pre',r.code));c.append(d);}if(r.file){const a=el('a','Open Python implementation');a.href=r.file;c.append(a);}if(r.refs.length){const p=el('p',undefined,'small');p.append(el('span','Original page scans: '));for(const ref of r.refs){const a=el('a',ref.label,'ref');a.href=ref.href;p.append(a);}c.append(p);}results.append(c);}}
q.addEventListener('input',render);kind.addEventListener('change',render);appendix.addEventListener('change',render);render();</script></html>'''
(ROOT/'index.html').write_text(template.replace('__SOURCES__',source_links).replace('__DATA__',data),encoding='utf-8')

inventory=['# Manual code inventory','', '38 identified example groups are mapped below. Full-page image links preserve original code exactly as visually printed. Raw OCR is included in records/manual_examples.jsonl and is NOT a verified verbatim transcription. Corrected equivalents change unsafe/incorrect examples and may change display styling or optional dependencies.','']
for g in groups:
    inventory += [f"## {g['id']} — {g['title']}",'',f"Corrected implementation: `{g['entry_point']}`; select `{g['output_selector']}`.",'',g['notes'],'']
    inventory += [f"- [{p['id']}]({p['image']})" for p in g['source_pages']]
    inventory += ['']
(ROOT/'MANUAL_CODE_INVENTORY.md').write_text('\n'.join(inventory).replace('38 identified',f'{len(groups)} identified'),encoding='utf-8')
print(json.dumps({'manual_example_groups':len(groups),'task_snippets':len(snippets),'qa_records':len(qa),'indexed_records':len(records),'sqlite_integrity':check}))
