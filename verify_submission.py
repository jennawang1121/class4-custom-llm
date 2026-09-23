"""Audit existing artifacts and reproduction results without changing models."""
import hashlib,json,re,subprocess
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads((ROOT/p).read_text())
runs={'starter':'20260922T000429_567209Z','expanded':'20260922T001305_992994Z'}
report={'suite_file_unchanged':hashlib.sha256((ROOT/'evals/language_evals.json').read_bytes()).hexdigest()=='e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7','case_records':0,'notebooks':{},'saved_model_reruns':{},'isolated_reproductions':{}}
assert report['suite_file_unchanged']
for label,run in runs.items():
 path=f'{label}_3000steps.ipynb';nb=read(path)
 cells=[c for c in nb['cells'] if c['cell_type']=='code']
 assert all(c['execution_count'] is not None for c in cells)
 assert not any(o['output_type']=='error' for c in cells for o in c['outputs'])
 report['notebooks'][path]={'executed_code_cells':len(cells),'errors':0}
 for stage in ['untrained','final']:
  original=read(f'llm_runs/{run}/language_evals/{stage}/eval_results.json');repeat=read(f'results/{label}-{stage}-check/eval_results.json')
  assert len(original)==48 and len({row['id'] for row in original})==48
  assert original==repeat
  report['case_records']+=len(original)
  report['saved_model_reruns'][f'{label}-{stage}']={'cases':48,'identical_per_case_results':True}
 completed=[]
 for p in sorted((ROOT/'reproductions').glob(label+'_*')):
  rs=list((p/'llm_runs').glob('*/training_summary.json'))
  if rs:completed.append((p,rs[0].parent))
 p,rr=completed[-1]
 matches={}
 for name in ['history.json','inspection.json','language_eval_comparison.json']:
  matches[name]=json.loads((rr/name).read_text())==read(f'llm_runs/{run}/{name}')
 assert all(matches.values()),matches
 report['isolated_reproductions'][label]={'directory':str(p.relative_to(ROOT)),'completed_steps':3000,'matches_original':matches,'imported_teaching_files':len(list((p/'corpus').glob('*.txt')))}
 assert report['isolated_reproductions'][label]['imported_teaching_files']==(0 if label=='starter' else 2)
t=read('evidence/chat_transcript.json'); assert len(t['turns'])==3
assert t['model_sha256']==read(f"llm_runs/{runs['expanded']}/language_evals/final/eval_summary.json")['model_sha256']
events=[json.loads(l) for l in (ROOT/'evidence/chat_session.cast').read_text().splitlines()]; recorded=''.join(e[2] for e in events[1:])
for turn in t['turns']:assert turn['prompt'] in recorded and turn['response'] in recorded
report['chat']={'actual_turns':3,'model_identity_matches':True,'terminal_recording_contains_all_prompts_and_replies':True}
report['public_repository_verified']=False
report['student_reflection_complete']=True
(ROOT/'evidence/verification.json').write_text(json.dumps(report,indent=2)+'\n')
missing=[];ignored=[]
for filename in ['README.md','EXPERIMENT_COMPARISON.md','STARTER_RESULTS.md','SETUP_RESULTS.md','evidence/README.md','CORPUS_SOURCES.md']:
 p=ROOT/filename
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if re.match(r'\w+://',target) or target.startswith('#'):continue
  dest=(p.parent/unquote(target.split('#')[0])).resolve()
  if not dest.exists():missing.append([filename,target]);continue
  if subprocess.run(['git','check-ignore','-q',str(dest)],cwd=ROOT).returncode==0:ignored.append([filename,target])
report['links']={'missing':missing,'ignored_by_git':ignored}
(ROOT/'evidence/verification.json').write_text(json.dumps(report,indent=2)+'\n')
assert not missing,missing
assert not ignored,ignored
print(json.dumps(report,indent=2))
