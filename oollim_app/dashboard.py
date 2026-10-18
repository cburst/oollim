"""Rich progress view inspired by the supplied supervisor dashboard."""
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from .common import STAGES

def dashboard(state):
 summary=Table.grid(expand=True);summary.add_column(ratio=1);summary.add_column(justify='right')
 summary.add_row(Text('oollim / speech evaluation',style='bold cyan'),Text(state['mode'],style='bold'))
 summary.add_row(Text(state['current'],overflow='ellipsis',no_wrap=True),Text(f"{state['family']} | CPU | {state['threads']} threads"))
 counts=Text(f"{state['done']} saved   {state['failed']} failed   {state['skipped']} reused   /   {state['total']} files",style='bold')
 table=Table(box=box.SIMPLE,expand=True,header_style='bold',padding=(0,1));table.add_column('PROCESS',ratio=3);table.add_column('STATUS',width=9);table.add_column('TIME',width=7,justify='right');table.add_column('DETAIL',ratio=4)
 for name,step in zip(STAGES,state['stages']):
  color={'done':'green','running':'cyan','failed':'red','skipped':'dim'}.get(step['status'],'dim');elapsed=step.get('elapsed',0)
  table.add_row(name,Text(step['status'].upper(),style=color),f'{int(elapsed)//60:02}:{int(elapsed)%60:02}' if step['status']!='waiting' else '--',Text(step.get('detail',''),overflow='ellipsis'))
 scores=state.get('scores',{});scoretext='   '.join(f'{d.title()}: {scores[d]:.2f}' if d in scores else f'{d.title()}: --' for d in ('accuracy','fluency','prosody'))
 activity=Group(*[Text(x) for x in state['activity'][-4:]])
 if state.get('error'):
  e=state['error'];activity=Group(Text(e['code']+': '+e['message'],style='red'),Text('Fix: '+e['fix'],style='yellow'),Text('Log: '+state.get('log',''),style='dim'))
 return Group(Panel(summary,border_style='cyan'),counts,table,Panel(Text(scoretext),title='Scores (raw 0–10 scale)'),Panel(activity,title='Needs attention' if state.get('error') else 'Recent activity',border_style='red' if state.get('error') else 'dim'),Text('Ctrl+C: stop safely | Completed results are saved | '+state['output'],style='dim'))
