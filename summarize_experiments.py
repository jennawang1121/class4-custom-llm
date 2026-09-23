"""Build an evidence report from saved measurements; never used for training."""
import json,hashlib
from pathlib import Path
runs={'Starter':Path('llm_runs/20260922T000429_567209Z'),'Expanded':Path('llm_runs/20260922T001305_992994Z')}
read=lambda p:json.loads(p.read_text())
assert hashlib.sha256(Path('evals/language_evals.json').read_bytes()).hexdigest()=='e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7'
lines=['# Two-experiment evidence report','','Both experiments trained fresh nanoGPT models for 3,000 steps at base learning rate 0.0015 with seed 42. The expanded experiment added 576 grammar and 445 spatial teaching passages. [Sources and design](CORPUS_SOURCES.md). No PDF sources or extraction warnings.','','These are public development tests. The test file and scoring are unchanged; no test prompts, keys, or outputs were added to teaching inputs. Vocabulary comes only from training passages.','','|Experiment / stage|All-case success|Scorable accuracy|Coverage|Evidence|','|---|---|---|---|---|']
for name,r in runs.items():
 for stage in ['untrained','final']:
  s=read(r/'language_evals'/stage/'eval_summary.json')['overall']; rows=read(r/'language_evals'/stage/'eval_results.json'); assert len(rows)==48
  lines.append(f"|{name} {stage}|{s['correct']}/48 ({s['success_rate_all_cases']:.2%})|{s['correct']}/{s['scorable']} ({s['accuracy_scorable_cases']:.2%})|{s['coverage']:.2%}|[JSON]({r}/language_evals/{stage}/eval_results.json), [CSV]({r}/language_evals/{stage}/eval_results.csv)|")
lines += ['', '## Interpretation','','Expanded final grammar is 3/3 and spatial relations 1/3. All six are newly scorable, but coverage alone is insufficient: two spatial cases remain wrong. The other 18 extension cases remain unscorable. The original 24 cases improve from 21/24 to 24/24, but changed vocabulary, initialization dimensions and data distribution prevent attributing this to a single cause. A shared seed does not yield identical weights when vocabulary dimensions change.','','A notable failure: lang_41 selects inside instead of below. Lang_42 selects north instead of right. These expose limitations in transferring single-sentence teaching to multi-sentence inverse-relation prompts. This is an interpretation, not a proven causal diagnosis.','','Four-choice success can hide poor free text: lang_26 correctly selects are, but its free continuation is "traffic discussion of course now ." Lang_40 selects book, yet its free continuation is only "." All results are retained.','','A next experiment could add varied multi-clause relational exercises with distinct stories, balancing directions and holding the public suite fixed. Unseen generalization would require additional untouched tests.','']
for name,r in runs.items():
 cfg=read(r/'config.json'); nb='starter_3000steps.ipynb' if name=='Starter' else 'expanded_3000steps.ipynb'; n=read(Path(nb)); assert all(c.get('execution_count') is not None for c in n['cells'] if c['cell_type']=='code'); assert not any(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[]))
 lines += [f'## {name}', '',f'[Executed notebook]({nb}) · [Complete results ZIP]({r}.zip)', '',f"Vocabulary: {cfg['vocabulary_size']}; parameters: {cfg['parameters']}; train/validation passages: {cfg['train_documents']}/{cfg['validation_documents']}; training/validation unknown-token rates: {cfg['training_unknown_rate']}/{cfg['validation_unknown_rate']}. Both rates are corpus measurements, distinct from test coverage.",'', '### Full fixed-panel loss table','','Each panel contains 20 documents. Panels remain fixed within a run; added data changes panels across runs. Losses across vocabularies are not directly comparable.','','|Step|Train|Validation|','|---|---|---|']
 for h in read(r/'history.json'): lines.append(f"|{h['step']}|{h['training_loss']}|{h['validation_loss']}|")
 lines += ['',f'![Training curves]({r}/training_curves.svg)','','### All category scores','','|Category|Before correct / scorable / total|After correct / scorable / total|','|---|---|---|']
 before=read(r/'language_evals/untrained/eval_summary.json'); after=read(r/'language_evals/final/eval_summary.json')
 for cat,a in before['by_category'].items():
  b=after['by_category'][cat];lines.append(f"|{cat}|{a['correct']} / {a['scorable']} / {a['total']}|{b['correct']} / {b['scorable']} / {b['total']}|")
 lines += ['','### Complete sample timeline','']
 for p in sorted((r/'samples').glob('*.txt')):lines += [f'[{p.stem}]({p})','','```text',p.read_text(),'```','']
 lines += ['### Inspections and reproducibility','']
 for f in ['config.json','corpus_manifest.json','vocabulary_report.json','split.json','tokenization.json','inspection.json','training.csv','training_summary.json','temperature_comparison.json','eval_separation.json']:
  lines.append(f'- [{f}]({r}/{f})')
lines += ['', '## Saved-model check','','The expanded model was loaded from model.pt and the unchanged runner reproduced 28/48. [Rerun summary](results/expanded_saved_model_check/eval_summary.json).','','```sh','.venv/bin/python run_evals.py --model llm_runs/20260922T001305_992994Z/model.pt --output results/new-eval-check','```','','The submission README now contains the learning explanations; evidence/ contains three actual chat interactions and a terminal recording. Student reflection and public publication remain pending.']
Path('EXPERIMENT_COMPARISON.md').write_text('\n'.join(lines)+'\n')
# Confirm the separately loaded checkpoint gives identical per-case results.
a=read(runs['Expanded']/'language_evals/final/eval_results.json');b=read(Path('results/expanded_saved_model_check/eval_results.json')); assert a==b
print('Verified 192 case records, both executed notebooks, unchanged suite, and identical saved-model rerun.')
