"""Export all notebook cells/outputs to GitHub-readable Markdown; no training."""
from pathlib import Path
import nbformat
import re
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence/notebook_exports';OUT.mkdir(exist_ok=True)
for experiment in ['starter','expanded']:
 path=ROOT/f'{experiment}_3000steps.ipynb';nb=nbformat.read(path,as_version=4)
 lines=[f'# {experiment.title()} executed notebook: complete readable export','',f'[Original executed notebook](../../{path.name})','','This is an additional reading format of the saved notebook, not a new run. It includes every cell and all saved outputs. The original notebook is the runnable artifact. HTML output is preserved as source below; SVG figures are exported without modification.','']
 for j,c in enumerate(nb.cells):
  lines += [f'## Cell {j+1}'+(f" · executed as {c.execution_count}" if c.cell_type=='code' else ''),'']
  if c.cell_type=='markdown':
   def relocate(m):
    target=m.group(1)
    return m.group(0) if re.match(r'[a-zA-Z]+:|#|/',target) else ']('+ '../../'+target+')'
   lines.extend([re.sub(r'\]\(([^)]+)\)',relocate,c.source),'']);continue
  lines += ['````python',c.source,'````','']
  for k,o in enumerate(c.get('outputs',[])):
   if o.output_type=='stream':lines += ['````text',o.text,'````','']
   elif o.output_type=='error':lines += ['````text','\n'.join(o.traceback),'````','']
   else:
    data=o.get('data',{})
    if 'image/svg+xml' in data:
     f=f'{experiment}_cell{j+1}_output{k+1}.svg';(OUT/f).write_text(data['image/svg+xml']);lines += [f'![Saved notebook figure]({f})','']
    if 'text/plain' in data:lines += ['````text',data['text/plain'],'````','']
    if 'text/html' in data:lines += ['<details><summary>Original HTML output source</summary>','','````html',data['text/html'],'````','','</details>','']
 (OUT/f'{experiment}.md').write_text('\n'.join(lines)+'\n')
 print(experiment,'exported',len(nb.cells),'cells; outputs preserved')
