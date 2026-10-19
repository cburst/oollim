"""Regenerate shared icons from the credited OpenMoji SVG (Qt + Pillow)."""
import io
from pathlib import Path
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PIL import Image
root=Path(__file__).resolve().parents[1]/'assets'
canvas=QImage(1024,1024,QImage.Format.Format_ARGB32)
canvas.fill(Qt.GlobalColor.transparent)
painter=QPainter(canvas);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
painter.setPen(Qt.PenStyle.NoPen);painter.setBrush(QColor('#BCE5FA'))
painter.drawRoundedRect(QRectF(32,32,960,960),210,210)
QSvgRenderer(str(root/'microphone.svg')).render(painter,QRectF(155,155,714,714))
painter.end();canvas.save(str(root/'oollim.png'))
image=Image.open(root/'oollim.png').convert('RGBA')
image.save(root/'oollim.ico',sizes=[(n,n) for n in (16,20,24,32,40,48,64,96,128,256)])
image.save(root/'oollim.icns')

# Native Windows wizard artwork, rendered from the same shared source.
for name,size,icon_size in [('installer-welcome.bmp',(164,314),132),('installer-header.bmp',(150,57),48)]:
    background=Image.new('RGB',size,'white')
    mark=image.resize((icon_size,icon_size),Image.Resampling.LANCZOS)
    background.paste(mark,((size[0]-icon_size)//2,(size[1]-icon_size)//2),mark)
    background.save(root/name)
