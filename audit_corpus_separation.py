"""Read-only corpus provenance audit. Writes a report outside all training inputs."""
import ast, hashlib, json, random, re, zipfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
suite=json.loads((ROOT/'evals/language_evals.json').read_text())
def tokens(s):return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]",s.lower(),flags=re.UNICODE)
def normalized(s):return ' '+' '.join(tokens(s))+' '
def matches(s):
 n=normalized(s)
 return [c['id'] for c in suite['cases'] if normalized(c['prompt']) in n]
# Extract only the original pure classroom-generation/tokenization functions.
tree=ast.parse((ROOT/'custom_llm.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['word_tokens','chunk_text','classroom_corpus']]
ns={'re':re};exec(compile(ast.Module(body=functions,type_ignores=[]),'<original classroom generator>','exec'),ns)
base=ns['chunk_text'](ns['classroom_corpus'](),47)
retained=[p for p in base if not matches(p)]
report={'scope':'Current source files, saved setup/formal corpora and splits, vocabularies, and submission ZIP','source_files':{},'runs':{},'issues':[],'limitations':'Normalized exact matching is not a semantic leakage detector. Generator templates were also manually reviewed. Ordinary shared vocabulary and underlying facts are allowed.'}
expected={'README.md','grammar.txt','spatial_relations.txt'}
actual={str(p.relative_to(ROOT/'corpus')) for p in (ROOT/'corpus').rglob('*') if p.is_file() or p.is_symlink()}
assert actual==expected,actual
extras=[]
for name in sorted(expected):
 p=ROOT/'corpus'/name;assert not p.is_symlink()
 text=p.read_text();hits=matches(text)
 assert not hits,(name,hits)
 markers=[x for x in ['predicted_choice','choice_probabilities','generated_text','eval_results','lang_01','model_sha256','answer key'] if x in text.lower()]
 assert not markers,(name,markers)
 report['source_files'][name]={'ingested':name!='README.md','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'reserved_prompt_matches':hits,'evaluation_metadata_markers':markers}
 if name!='README.md':extras+=ns['chunk_text'](text,47)
for label,run in [('setup','20260921T224253_008282Z'),('starter','20260922T000429_567209Z'),('expanded','20260922T001305_992994Z')]:
 r=ROOT/'llm_runs'/run;raw=(r/'corpus.txt').read_text();expected_raw='\n'.join(retained+(extras if label=='expanded' else []))
 assert raw==expected_raw,(label,'unexpected corpus content')
 assert not matches(raw)
 split=json.loads((r/'split.json').read_text());docs=sorted(set(expected_raw.splitlines()));random.Random(42).shuffle(docs);cut=int(.9*len(docs))
 assert split['train']==docs[:cut] and split['validation']==docs[cut:]
 counts=Counter(t for doc in split['train'] for t in tokens(doc));selected=sorted((t for t in counts if len(t)<=128),key=lambda t:(-counts[t],t))[:509]
 vocab=['<UNK>','<BOS>','<EOS>']+sorted(selected)
 assert vocab==json.loads((r/'tokenization.json').read_text())['vocabulary']
 manifest=json.loads((r/'corpus_manifest.json').read_text())
 for f in manifest['files']:assert f['sha256']==report['source_files'][f['file']]['sha256']
 with zipfile.ZipFile(ROOT/'llm_runs'/f'{run}.zip') as z:assert z.read('corpus.txt').decode()==raw
 report['runs'][label]={'corpus_matches_only_original_classroom_and_expected_teaching_files':True,'reserved_prompt_matches':[],'saved_split_exactly_reconstructed':True,'vocabulary_exactly_reconstructed_from_training_only':True,'imported_files':[f['file'] for f in manifest['files']],'excluded_reserved_passages':len(base)-len(retained),'run_zip_corpus_matches':True}
bundle=ROOT.parent/'class4-submission.zip'
with zipfile.ZipFile(bundle) as z:
 members=[n for n in z.namelist() if '/corpus/' in n and not n.endswith('/')]
 assert sorted(Path(n).name for n in members)==sorted(expected),members
 for n in members:assert z.read(n)==(ROOT/'corpus'/Path(n).name).read_bytes()
 for label,run in [('starter','20260922T000429_567209Z'),('expanded','20260922T001305_992994Z')]:
  assert z.read(f'class4-custom-llm/llm_runs/{run}/corpus.txt')==(ROOT/'llm_runs'/run/'corpus.txt').read_bytes()
 report['submission_zip']={'corpus_members':members,'matches_audited_local_files':True}
report['manual_review']='Reviewed all extension generator templates: agreement/tense sentences and independent spatial descriptions; no answer-key lists, scored responses, or copied multi-sentence test stories. Shared ordinary words and grammar/spatial concepts are intentional teaching content.'
(ROOT/'evidence/corpus_separation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
