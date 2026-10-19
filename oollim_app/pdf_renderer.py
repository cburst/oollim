"""WeasyPrint rendering in an isolated process, with no external URL access."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--landscape',action='store_true');args=parser.parse_args()
    from weasyprint import HTML
    def reject_url(url,*args,**kwargs):raise ValueError('External resources are disabled in local speech reports.')
    html=args.input.read_text(encoding='utf-8')
    if args.landscape:html=html.replace('size:A4;','size:A4 landscape;')
    HTML(string=html,url_fetcher=reject_url).write_pdf(args.output)

if __name__=='__main__':main()
