"""Heatmapper-compatible local PDF, with WeasyPrint and no email/network use."""
from html import escape
import math
import os
from pathlib import Path
import tempfile
from .report_templates import heatmap_html, word_heatmap_html


def score_color(score):
    return '#b7e4c7' if score>8.5 else '#ffe8a1' if score>8 else '#f7c6a3' if score>7.5 else '#f5a3a3'


def word_color(score):
    return '#b7e4c7' if score>9.5 else '#ffe8a1' if score>9 else '#f7c6a3' if score>8 else '#f5a3a3'


def table_html(rows,title,landscape=False):
    headers=list(rows[0])
    head=''.join('<th>'+escape(h.replace('_',' '))+'</th>' for h in headers)
    if 'prosody_score' in headers:
        averages={}
        for name in ('accuracy_score','fluency_score','prosody_score'):
            values=[float(r[name]) for r in rows if isinstance(r.get(name),(float,int))]
            averages[name]=round(sum(values)/len(values),2) if values else ''
        rows=[*rows,{name:('AVERAGE' if i==0 else averages.get(name,'')) for i,name in enumerate(headers)}]
    body=''.join('<tr>'+''.join('<td>'+escape(str(row[h]))+'</td>' for h in headers)+'</tr>' for row in rows)
    return '''<html><head><style>@page {size:A4; margin:1in;}
    body {font-family:Arial,sans-serif; font-size:10pt;}
    table {width:100%; border-collapse:collapse; page-break-inside:avoid;}
    th,td {border:1px solid black; padding:5px; text-align:left; vertical-align:top;}
    th {background-color:#f2f2f2;} thead {display:table-header-group;}
    tr {page-break-inside:avoid;} .section-title-header {font-weight:bold; font-size:12pt; background-color:#e0e0e0; text-align:left; padding:5px;}
    </style></head><body><table><thead><tr><th class="section-title-header" colspan="'''+str(len(headers))+ '">'+escape(title)+'</th></tr><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></body></html>'


def report_documents(row,reference,granular=None):
    if row.get('status')!='completed' or not all(math.isfinite(float(row[k])) for k in ('accuracy','fluency','prosody')):
        raise ValueError('A PDF requires a completed recording with three finite scores.')
    sentences=granular['sentences'] if granular else [dict(text=reference,accuracy=float(row['accuracy']),fluency=float(row['fluency']),prosody=float(row['prosody']),start_seconds=0,end_seconds=row.get('duration_seconds',''))]
    paragraph=' '.join('<mark class="hl" style="background:'+ (score_color(s['prosody']) if s.get('prosody') is not None else '#dddddd')+';">'+escape(s['text'])+'</mark>' for s in sentences)
    documents=[(heatmap_html(paragraph),False)]
    if granular:
        words=[];cursor=0
        for word in granular['words']:
            words.append(escape(reference[cursor:word['char_start']]))
            words.append('<span class="hl-word" style="background:'+word_color(word['accuracy'])+';">'+escape(reference[word['char_start']:word['char_end']])+'</span>')
            cursor=word['char_end']
        words.append(escape(reference[cursor:]))
        documents.append((word_heatmap_html(''.join(words)),False))
    sentence_rows=[dict(sentence_index=i,reference_text=s['text'],**{name+'_score':round(10*s[name],2) if s.get(name) is not None else 'Unavailable' for name in ('accuracy','fluency','prosody')}) for i,s in enumerate(sentences)]
    documents.append((table_html(sentence_rows,'Sentence Level Data'),True))
    if granular and granular['words']:
        words=[dict(sentence_index=w['sentence_index'],word=w['text'],accuracy_score=round(10*w['accuracy'],2)) for w in granular['words']]
        documents.append((table_html(words,'Word Level Data'),False))
    return documents


def report_html(row,reference,granular=None):
    return '\n'.join(html for html,_ in report_documents(row,reference,granular))


def render_html(html,destination,landscape):
    import subprocess,sys
    from .common import ROOT
    html=html.replace('<head>','<head><meta charset="utf-8">',1)
    source=Path(destination).with_suffix('.html');source.write_text(html,encoding='utf-8')
    from .pdf_runtime import environment,windows_exe
    if landscape:source.write_text(html.replace('size:A4;','size:A4 landscape;'),encoding='utf-8')
    command=[sys.executable,'-m','oollim_app.pdf_renderer','--input',str(source),'--output',str(destination)]
    if os.name=='nt':command=[str(windows_exe()),'--encoding','utf-8',str(source),str(destination)]
    elif landscape:command.append('--landscape')
    try:
        result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45,env=environment())
        if result.returncode or not Path(destination).exists():
            raise OSError('PDF rendering failed: '+result.stderr.decode(errors='replace')[-1000:])
    finally:source.unlink(missing_ok=True)


def write_report(path,row,reference,granular=None):
    from pypdf import PdfWriter
    documents=report_documents(row,reference,granular)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.oollim-pdf-',dir=path.parent) as temp:
        writer=PdfWriter()
        try:
            for i,(html,landscape) in enumerate(documents):
                part=Path(temp)/f'{i}.pdf';render_html(html,part,landscape);writer.append(part)
            merged=Path(temp)/'analysis.pdf'
            with merged.open('wb') as stream:writer.write(stream)
            os.replace(merged,path)
        finally:writer.close()
    return path
