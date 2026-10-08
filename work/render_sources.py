from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work'/'deps'))
import pypdfium2 as pdfium
from PIL import Image,ImageOps,ImageDraw
OUT=ROOT/'outputs'/'cv_knowledge_base'
PAGES=OUT/'extracted'/'page_images'; PAGES.mkdir(parents=True,exist_ok=True)
CONTACT=ROOT/'work'/'contacts'; CONTACT.mkdir(parents=True,exist_ok=True)
jobs=[]
for source in json.loads((OUT/'records'/'sources.json').read_text()):
    doc=pdfium.PdfDocument(OUT/source['file'])
    thumbs=[]
    for i,page in enumerate(doc,1):
        im=page.render(scale=2.3).to_pil().convert('RGB')
        if max(im.size)>3800: im.thumbnail((3800,3800))
        pid=f'{source["id"]}_p{i:03}'
        path=PAGES/f'{pid}.jpg'; im.save(path,quality=90)
        jobs.append(dict(id=pid,source_id=source['id'],page=i,path=str(path.resolve())))
        th=ImageOps.contain(im,(470,650))
        card=Image.new('RGB',(490,690),'white'); card.paste(th,((490-th.width)//2,25))
        ImageDraw.Draw(card).text((8,5),f'{source["id"]} page {i}',fill='black')
        thumbs.append(card)
    for start in range(0,len(thumbs),6):
        sheet=Image.new('RGB',(1470,1380),'#cccccc')
        for j,th in enumerate(thumbs[start:start+6]): sheet.paste(th,((j%3)*490,(j//3)*690))
        sheet.save(CONTACT/f'{source["id"]}_{start+1:03}.jpg',quality=90)
    print(source['id'],len(doc),flush=True)
(ROOT/'work'/'ocr_jobs.json').write_text(json.dumps(jobs),encoding='utf-8')
