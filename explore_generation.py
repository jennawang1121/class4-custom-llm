"""Small inference-only exploration; no training, vocabulary changes, or eval edits."""
import csv, hashlib, json
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import torch
from run_evals import load_model, model_hash, generate_reply, word_tokens, normalized
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'llm_runs/20260922T001305_992994Z/model.pt'
OUT=ROOT/'evidence/additional_exploration'
SEEDS=[2026,2027,2028,2029,2030]
# Defined before generating any outputs; no selection based on favorable results.
PROMPTS=['last monday she walked to the','she walked last monday to the']
TEMPERATURES=[0.3,0.8,1.2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 protected=[MODEL,ROOT/'evals/language_evals.json',*sorted((ROOT/'corpus').glob('*.txt'))]
 before={str(p.relative_to(ROOT)):sha(p) for p in protected}
 protocol={'model':str(MODEL.relative_to(ROOT)),'seeds':SEEDS,'max_tokens':24,'fresh_context_per_prompt':True,'phrasing':{'prompts':PROMPTS,'temperature':0.8,'intent':'Same subject, past walking event and destination slot; reposition last monday.'},'temperature':{'prompt':'the customer','values':TEMPERATURES},'limitations':['Two word orders can change token positions and local context as well as familiarity.','Five seeds per condition are exploratory, not an accuracy benchmark.','Unique outputs and word counts measure diversity, not correctness.'],'prediction':'Word order may affect continuation despite known words. Higher temperature may increase variation but can also preserve the same output or generate less coherent text.'}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 suite=json.loads((ROOT/'evals/language_evals.json').read_text())
 for prompt in PROMPTS+['the customer']:
  assert all(normalized(c['prompt']) not in normalized(prompt) for c in suite['cases'])
 torch.set_num_threads(4)
 model,vocabulary,saved=load_model(MODEL);identity=model_hash(model)
 rows=[]
 for experiment,prompts,temps in [('phrasing',PROMPTS,[0.8]),('temperature',['the customer'],TEMPERATURES)]:
  for prompt in prompts:
   for temperature in temps:
    for seed in SEEDS:
     reply=generate_reply(model,vocabulary,prompt,seed=seed,temperature=temperature,max_tokens=24)
     rows.append({'experiment':experiment,'prompt':prompt,'temperature':temperature,'seed':seed,**reply})
 assert model_hash(model)==identity
 assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
 (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (OUT/'results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
  for row in rows:writer.writerow({**row,'unknown_prompt_words':json.dumps(row['unknown_prompt_words'])})
 summary=[]
 for experiment,prompts,temps in [('phrasing',PROMPTS,[0.8]),('temperature',['the customer'],TEMPERATURES)]:
  for prompt in prompts:
   for temperature in temps:
    group=[r for r in rows if r['experiment']==experiment and r['prompt']==prompt and r['temperature']==temperature]
    summary.append({'experiment':experiment,'prompt':prompt,'temperature':temperature,'samples':len(group),'distinct_full_continuations':len({r['response'] for r in group}),'distinct_generated_token_types':len({t for r in group for t in word_tokens(r['response'])}),'empty_responses':sum(not r['response'] for r in group),'unknown_prompt_words':sorted({w for r in group for w in r['unknown_prompt_words']})})
 audit={'model_identity':identity,'completed_training_steps':saved['completed_steps'],'protected_file_hashes_before_and_after':before,'model_weights_unchanged':True,'protected_files_unchanged':True,'fixed_eval_prefixes_not_used':True,'generated_responses':len(rows),'timestamp_utc':datetime.now(timezone.utc).isoformat()}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
 for row in rows:print(row['experiment'],row['temperature'],row['seed'],row['prompt'],'=>',row['response'])
if __name__=='__main__':main()
