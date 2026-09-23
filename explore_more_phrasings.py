"""Two additional paired wording comparisons; inference only."""
import json,hashlib,csv
from pathlib import Path
import torch
from run_evals import load_model,model_hash,generate_reply,normalized
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence/additional_phrasings'
PAIRS={'teacher_state':['today the teacher is','the teacher today is'],'object_location':['a pencil is inside the','inside the basket is a']}
# Location prompts describe containment, but leave different slots to complete.
# This intentionally tests a different syntactic formulation, not identical targets.
SEEDS=[2026,2027,2028,2029,2030]
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 protocol={'pairs':PAIRS,'seeds':SEEDS,'temperature':0.8,'max_tokens':24,'model':'llm_runs/20260922T001305_992994Z/model.pt','prediction':'Teacher word-order changes may preserve continuation. Inverted location wording may produce different completions or reveal reliance on familiar templates.','limitations':['The teacher pair changes word order while retaining the same completion slot.','The location pair changes the open slot: A needs a container; B needs an object. Different output is expected and is not by itself a failure.','Five paired seeds per prompt are qualitative exploration, not a benchmark.']}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 protected=[ROOT/protocol['model'],ROOT/'evals/language_evals.json',*sorted((ROOT/'corpus').glob('*.txt'))]
 hashes=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
 before=hashes();suite=json.loads((ROOT/'evals/language_evals.json').read_text())
 for pair in PAIRS.values():
  for p in pair: assert all(normalized(c['prompt']) not in normalized(p) for c in suite['cases'])
 torch.set_num_threads(4);model,vocab,_=load_model(ROOT/protocol['model']);identity=model_hash(model)
 rows=[]
 for group,pair in PAIRS.items():
  for variant,prompt in zip(['A','B'],pair):
   for seed in SEEDS:
    rows.append({'group':group,'variant':variant,'prompt':prompt,'seed':seed,'temperature':0.8,**generate_reply(model,vocab,prompt,seed=seed,temperature=0.8,max_tokens=24)})
 assert before==hashes() and model_hash(model)==identity
 (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (OUT/'results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({**r,'unknown_prompt_words':json.dumps(r['unknown_prompt_words'])})
 (OUT/'audit.json').write_text(json.dumps({'model_identity':identity,'weights_unchanged':True,'protected_files_unchanged':True,'file_hashes':before,'total_responses':len(rows),'fixed_eval_prefixes_not_used':True},indent=2)+'\n')
 for r in rows: print(r['group'],r['variant'],r['seed'],r['prompt'],'=>',r['response'],'unknown',r['unknown_prompt_words'])
if __name__=='__main__':main()
