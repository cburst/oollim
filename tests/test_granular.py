import numpy as np
import pytest
from oollim_app.granular import align_ctc, sentence_spans


def test_repeated_phone_requires_blank():
    p=np.log(np.array([[.01,.99],[.99,.01],[.01,.99]]))
    result=align_ctc(p,[1,1],0)
    assert [x.tolist() for x in result]==[[0],[2]]
    with pytest.raises(ValueError,match='cannot be aligned'):align_ctc(p[:2],[1,1],0)


def test_reference_phone_order():
    p=np.log(np.array([[.01,.98,.01],[.98,.01,.01],[.01,.01,.98]]))
    assert [x.tolist() for x in align_ctc(p,[1,2],0)]==[[0],[2]]


def test_nltk_sentence_character_spans():
    text='Dr. Rose spoke. Please call Stella!'
    assert [text[a:b] for a,b in sentence_spans(text)]==['Dr. Rose spoke.','Please call Stella!']


def test_low_phone_support_is_not_promoted_to_high_accuracy():
    p=np.log(np.array([[.999,.001],[.999,.001],[.999,.001]]))
    frames=align_ctc(p,[1],0)[0]
    assert 10*np.exp(p[frames,1]).mean()==pytest.approx(.01)


def test_pdf_reports_segments_and_escapes_reference(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication
    from oollim_app.report import report_html,write_report
    app=QApplication.instance() or QApplication([])
    row=dict(audio='sample.wav',status='completed',family='speechocean',model_version='test',accuracy=8,fluency=8,prosody=8)
    detail=dict(words=[dict(text='Test',char_start=0,char_end=4,accuracy=7,sentence_index=0,start_seconds=0,end_seconds=1)],sentences=[dict(text='Test <text>.',prosody=8,start_seconds=0,end_seconds=1)])
    html=report_html(row,'Test <text>.',detail)
    assert 'start seconds' not in html and 'end seconds' not in html
    assert all(name+' score' in html for name in ('accuracy','fluency','prosody'))
    assert '&lt;text&gt;' in html and 'experimental' in html and 'Speech Quality and Color Scoring System:' in html
    def fake_render(html,destination,landscape):
        from pypdf import PdfWriter
        writer=PdfWriter();writer.add_blank_page(width=595,height=842)
        with open(destination,'wb') as handle:writer.write(handle)
    monkeypatch.setattr('oollim_app.report.render_html',fake_render)
    path=write_report(tmp_path/'report.pdf',row,'Test <text>.',detail)
    assert path.read_bytes().startswith(b'%PDF')
    row['accuracy']=float('nan')
    before=path.read_bytes()
    with pytest.raises(ValueError):write_report(path,row,'text')
    assert path.read_bytes()==before


def test_original_heatmap_thresholds_and_no_contact():
    from oollim_app.report import score_color,word_color
    from oollim_app.report_templates import heatmap_html,word_heatmap_html
    assert score_color(8.5)=='#ffe8a1'
    assert word_color(9.5)=='#ffe8a1'
    assert word_color(8.5)=='#f7c6a3'
    for html in [heatmap_html('Example.'),word_heatmap_html('Example.')]:
        assert 'Contact info' not in html and '@hufs' not in html
        assert 'border-radius:12px' in html and 'font-size: 14pt' in html
        assert 'Microsoft Azure' not in html
