"""Assemble the README from original saved evidence, never inventing measurements."""
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads((ROOT/p).read_text())
runs={'Starter':'llm_runs/20260922T000429_567209Z','Expanded':'llm_runs/20260922T001305_992994Z'}
L=[]
def add(s=''):L.append(s)
add('''# Class 4: training and evaluating a tiny nanoGPT

Two fresh nanoGPT models were trained on CPU for **3,000 steps at learning rate 0.0015**: first on the supplied classroom corpus, then on the classroom corpus plus original grammar and spatial-relation teaching text. The expanded model scored **28/48** on the unchanged public development tests, versus **21/48** for the trained starter. It still failed two spatial questions and could not score 18 other extension questions because of missing vocabulary. It is a narrow sentence-completion model, not a general chatbot.

This report was prepared with AI assistance from actual notebook outputs. The student selected the settings and extension categories. Explanations below are evidence-based drafts for student review; they are not a claim that the student has already supplied a personal reflection. [Learning checkpoints](REFLECTION.md) remain for the student to answer in their own words.

## Start with the evidence

- [Executed starter notebook](starter_3000steps.ipynb) and [executed expanded notebook](expanded_3000steps.ipynb), with outputs retained.
- [Detailed experiment comparison and all saved sample timelines](EXPERIMENT_COMPARISON.md).
- [Actual chat transcript, recording, and launch instructions](evidence/README.md).
- [Fixed 48-case suite](evals/language_evals.json) and [unchanged evaluation runner](run_evals.py).
- [Original teaching sources and permissions](CORPUS_SOURCES.md), [grammar text](corpus/grammar.txt), [spatial text](corpus/spatial_relations.txt), and [reproducible generator](build_extension.py).
- [Reproduction and evidence checks](evidence/verification.json).

## Choices, prediction, and outcomes

**Corpus:** the first experiment used only the supplied synthetic classroom sentences. The second added grammar (agreement, pronouns, and tense) and spatial relations (containment, vertical/horizontal location, and direction). Both categories had no scorable cases under the starter vocabulary. The extension contains 576 unique grammar passages and 445 unique spatial passages, all original synthetic text drafted with AI assistance for this assignment. No private records or third-party documents were imported. The teaching files can be shared; no PDF extraction was needed, and the manifests report no import warnings.

Examples of teaching material include `last monday she walked to the garden .` and `below a clock stands a chair .`. The generator never reads the test suite, answer keys, or evaluation outputs. It provides varied independent teaching sentences rather than copied test stories or answer lists. Ordinary words and underlying concepts overlap with the public benchmark by design.

**Steps:** a 10-step setup run checked the pipeline; 3,000 updates were chosen for each formal run, following the instructor's suggested starting budget. A step is one batch-based parameter update, not a full pass through the corpus.

**Learning rate:** the student selected 0.0015 rather than the starter's 0.001. This is 50% larger, but no controlled learning-rate comparison was run, so no claim is made that it is better. Too-large updates can destabilize learning; too-small updates can make progress slow. The notebook uses warmup and cosine decay, so the actual learning rate varies. AdamW is not simply a fixed learning-rate-times-gradient update.

**Prediction, recorded before training:** loss might decrease, samples might resemble classroom sentences more closely, and broader teaching material might improve coverage and selected skills. Better four-choice scores or fluent text were not guaranteed. The prediction cells remain in both executed notebooks.

**Observed:** loss fell within both runs and the saved timeline samples became recognizable classroom-style sentences. The expanded run covered six additional tests; grammar scored 3/3, spatial relations 1/3. Some correct four-choice selections still had incoherent free continuations. These findings partly support the predictions, without establishing general language understanding.

**Controls:** both formal runs kept 3,000 steps, base learning rate 0.0015, seed 42, batch size 32, two blocks, four attention heads, 64-dimensional embeddings, 48-token context, the split procedure, scoring, and generation settings. Added text changed the vocabulary, data distribution, actual split/panel membership, and parameter count. The same seed does not imply identical initial weights when tensor sizes change. Loss values across different vocabularies/corpora are not directly comparable. Within each run, the loss panels and generation settings remained fixed.

## Four complete evaluations

Only the prompt enters nanoGPT. The runner compares probabilities for four possible next words afterward: the correct word must be strictly highest; ties and incorrect choices score zero. Unknown prompt/choice words or excessive context make a case unscorable and give zero in the all-case metric. No failures are dropped. Scorable accuracy has a different denominator and must be read with coverage.

Free continuations are generated separately (temperature 0.8, fixed per-case seed, up to 24 tokens); they are not the four-choice score. The suite is a **public development benchmark**, because its categories informed the extension. It is not an untouched test of unseen generalization.

| Experiment | Stage | Correct / 48 | Scorable / 48 (coverage) | Scorable accuracy | Complete evidence |
|---|---|---|---|---|---|''')
for label,r in runs.items():
 for stage in ['untrained','final']:
  s=read(f'{r}/language_evals/{stage}/eval_summary.json')['overall'];base=f'{r}/language_evals/{stage}'
  add(f"|{label}|{stage}|{s['correct']}/48 ({s['success_rate_all_cases']:.2%})|{s['scorable']}/48 ({s['coverage']:.2%})|{s['accuracy_scorable_cases']:.2%}|[CSV]({base}/eval_results.csv) · [JSON]({base}/eval_results.json) · [summary]({base}/eval_summary.json)|")
add('\n### Group and category results\n\nEntries are **correct / scorable / total**. Zero scorable cases means missing coverage, not a measured 0% accuracy among scorable cases.\n')
for grouping in ['by_group','by_category']:
 add('|'+('Group' if grouping=='by_group' else 'Category')+'|Starter before|Starter after|Expanded before|Expanded after|\n|---|---|---|---|---|')
 summaries=[read(f'{r}/language_evals/{stage}/eval_summary.json') for r in runs.values() for stage in ['untrained','final']]
 for key in summaries[0][grouping]:
  add('|'+key+'|'+'|'.join(f"{s[grouping][key]['correct']} / {s[grouping][key]['scorable']} / {s[grouping][key]['total']}" for s in summaries)+'|')
 add()
add('''The trained-model gain of seven correct cases comprises three additional starter-transfer successes plus four newly covered extension successes. The six new scorable cases are the three grammar and three spatial questions. Their expanded untrained score was 0/6, and trained score was 4/6; this is consistent with learning beyond vocabulary inclusion, but does not isolate a causal mechanism. Vocabulary size and initialization also changed between experiments.

Concrete limitations in the expanded run:

- `lang_41` selected `inside` rather than `below`; `lang_42` selected `north` rather than `right`. Knowing the words did not guarantee inverse spatial reasoning.
- `lang_26` correctly ranked `are` among four choices, but freely continued with `traffic discussion of course now .`.
- `lang_40` ranked `book` highest among its four choices, but its free continuation was only `.`. Its probability for `book` was only about 0.0057% of the whole vocabulary. Being best among four candidates does not make a word likely overall.
- The other 18 extension cases remain unscorable. No claim is made that this model has mastered those skills.

The full case records linked above preserve prompts, choice probabilities, status, selected answers, and actual continuations, including failures.

### Separation from training

The suite stayed in `evals/`; only teaching files were placed in `corpus/`. Test prompts, paired reference answers, scoring rules, evaluation outputs, and chat transcripts were never used to build the vocabulary or update weights. Each run excluded 160 generated classroom passages containing reserved test prefixes before splitting/vocabulary construction. Imported files passed the supplied normalized-prefix check. The suite's original file SHA-256 remains `e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7` (the runner also reports a separately computed canonical suite hash). Exact matching is not a semantic leakage detector; the added teaching text was also reviewed for copied test stories and answer lists.
''')
for label,r in runs.items():add(f'- {label}: [separation check]({r}/eval_separation.json) · [file manifest]({r}/corpus_manifest.json) · [training text]({r}/corpus.txt).')
add('\n## Run settings, data, and complete loss history\n\nThe 90/10 split is by deduplicated passage, not source file. Training and held-out text share templates, so held-out loss tests combinations within this restricted distribution. It does not establish transfer to new domains or unseen templates. Vocabulary is built only from the training split, retaining at most 509 ordinary types plus UNK/BOS/EOS; neither formal run reached that limit.\n')
add('|Run|Completed steps|Training-loop seconds|Parameters|Vocabulary|Train / validation passages|Train / held-out UNK rate|\n|---|---|---|---|---|---|---|')
for label,r in runs.items():
 c=read(f'{r}/config.json');s=read(f'{r}/training_summary.json')
 add(f"|{label}|{s['completed_steps']}|{s['elapsed_seconds']:.6f}|{c['parameters']:,}|{c['vocabulary_size']}|{c['train_documents']} / {c['validation_documents']}|{c['training_unknown_rate']:.1%} / {c['validation_unknown_rate']:.1%}|")
add('\nBoth used CPU on macOS 15.7.7 arm64, Python 3.12.14 and PyTorch 2.14.0. Timings are the original notebook training-loop measurements, not download/setup or total notebook wall time. Neither formal run was interrupted. An initial setup attempt failed before training because NumPy was absent; NumPy was installed and the notebook rerun from the beginning. The successful 10-step setup is [preserved separately](setup_10steps.ipynb); the incomplete attempt is excluded from submission evidence.\n')
for label,r in runs.items():
 add(f'### {label} loss\n\nEach fixed panel has **20 documents** (the notebook supports at most 20). Loss is averaged over non-padding next-token targets, not all corpus text. Every measured value is below.\n\n|Step|Training loss|Validation loss|\n|---|---|---|')
 for h in read(f'{r}/history.json'):add(f"|{h['step']}|{h['training_loss']}|{h['validation_loss']}|")
 add(f'\n![{label} loss]({r}/training_curves.svg)\n\n[history.json]({r}/history.json) · [training.csv]({r}/training.csv) · [config]({r}/config.json) · [training summary]({r}/training_summary.json) · [vocabulary report]({r}/vocabulary_report.json)\n')
add('## Untrained, halfway, and final samples\n\nEvery saved sample is included below, including garbled text. Each stage uses BOS, sampling seed 2026, temperature 0.8, four samples, and a 32-token limit. This sample timeline differs from the 24-token evaluation continuations.\n')
for label,r in runs.items():
 for step in [0,1500,3000]:
  path=f'{r}/samples/step_{step:04d}.txt';add(f'### {label}: step {step}\n\n[Complete sample file]({path})\n\n```text\n{(ROOT/path).read_text()}\n```\n')
add('The untrained samples are mostly disconnected words. Halfway and final samples largely use the supplied classroom templates. Some samples are unchanged across stages; the complete timeline above preserves this. Plausible template text is narrower evidence than broad language understanding.\n')
add('''## How learning works, using recorded values

A **corpus** is the teaching text. A **token** is the unit processed by this notebook: a whole word or punctuation mark after normalization. A token ID is an arbitrary lookup index. A **vector** is an ordered list of numbers; an **embedding** is a trainable vector assigned to a token. Its coordinates are not pre-labeled human concepts.

In the starter run, `customer` maps to **ID 28**, which selects a row of **64 numbers**. In the expanded vocabulary it maps to **ID 50**, illustrating that IDs depend on the vocabulary and are not meanings. Each ID stays fixed within its own run. [Starter tokenization](llm_runs/20260922T000429_567209Z/tokenization.json), [expanded tokenization](llm_runs/20260922T001305_992994Z/tokenization.json).

The starter's complete recorded `customer` vectors are shown below (rounded to nine decimal places; the inspection JSON retains full precision).
''')
i=read(runs['Starter']+'/inspection.json')
for key in ['embedding_before','embedding_after']:
 add(f'**{key} (64 coordinates):**\n\n```text')
 for k in range(0,64,8):add(', '.join(f'{x:.9f}' for x in i[key][k:k+8]))
 add('```\n')
add('''A neural network combines many learned weights: token/position embeddings, attention projections, and feed-forward transformations, with nonlinear GELU operations, normalization, and residual connections. Its final scores become a probability distribution via softmax. Attention mixes information from earlier tokens and the current token; the causal mask blocks future positions so the model cannot peek at the word it must predict. The notebook's recorded first-head attention is one part of the actual computation, not a universal explanation of the model.

For next-token learning, the loss penalizes low probability on the observed next word. PyTorch backpropagation computes **gradients**: how the current loss would locally change if a parameter changed slightly. AdamW uses them, its running statistics, weight decay, and the learning-rate schedule to update weights. The embedding is one set of these trainable parameters.

### One real gradient and first update
''')
add('|Run / customer coordinate 0|Before|Gradient (before clipping)|Actual first-step learning rate|After|Change|\n|---|---|---|---|---|---|')
for label,r in runs.items():
 f=read(r+'/inspection.json')['first_update'];add(f"|{label}|{f['before']}|{f['gradient']}|{f['learning_rate']}|{f['after']}|{f['after']-f['before']:.12g}|")
add('\nWarmup makes the first update use **0.000015**, although the configured base rate is 0.0015. In the starter example the recorded gradient is positive and this coordinate decreases slightly. That is one actual update, distinct from the much larger net change after 3,000 steps. The gradient is recorded before gradient clipping; the code clips the overall norm before AdamW updates. These values should not be interpreted as a simple SGD calculation.\n')
add('### A next-token probability change\n\nFor the same prefix `the customer`, in the starter run:\n\n|Candidate|Before training|After training|\n|---|---|---|')
v=read(runs['Starter']+'/tokenization.json')['vocabulary']
for word in ['reviewed','compared','customer']:
 k=v.index(word);add(f"|{word}|{i['probabilities_before'][k]:.6%}|{i['probabilities_after'][k]:.6%}|")
add('\nThese changes come from the whole trained network, not just one embedding coordinate. Generation samples a token from the distribution, appends it to the context, and repeats until EOS or the output limit. A rising correct-word probability can reduce loss without changing which of four candidates ranks first; this helps explain why loss and benchmark accuracy need not move together.\n')
for label,r in runs.items():add(f'- [{label}: full vectors, gradients, probabilities, and attention]({r}/inspection.json) · [viewer checkpoint]({r}/checkpoint.json).')
add('\n### Embedding neighbors\n\nCosine similarity below uses all 64 dimensions of the starter embeddings, excludes the token itself, and was computed from the saved checkpoint. It is not a test score.\n')
ck=read(runs['Starter']+'/checkpoint.json');target=ck['vocabulary'].index('customer')
def neighbors(matrix):
 a=matrix[target];na=math.sqrt(sum(x*x for x in a));scores=[]
 for j,b in enumerate(matrix):
  if j==target:continue
  nb=math.sqrt(sum(x*x for x in b));scores.append((sum(x*y for x,y in zip(a,b))/(na*nb),ck['vocabulary'][j]))
 return sorted(scores,reverse=True)[:5]
for label,matrix in [('Before',ck['initial_embeddings']),('After',ck['weights']['wte'])]:add(f"- {label}: "+', '.join(f'{w} ({score:.3f})' for score,w in neighbors(matrix))+'.')
add('\nThe neighbors reflect the restricted contexts in the teaching sentences, not human-defined semantic labels. To inspect visually, open [embedding-viewer.html](embedding-viewer.html) locally and load the desired checkpoint.json. PCA compresses 64 dimensions for display and can distort apparent distances; cosine neighbors use the full vectors. The viewer is not a chat interface.\n')
add('## Temperature: inference without retraining\n\nTemperature divides output scores before softmax. Lower temperature concentrates probability on favored tokens; higher temperature flattens it. It changes sampling, not vocabulary or learned weights. The notebook resets sampling seed 2026 and starts at BOS for each temperature; all saved samples follow.\n')
for label,r in runs.items():
 add(f'### {label}\n\n[Full temperature comparison]({r}/temperature_comparison.json)\n')
 for t,samples in read(r+'/temperature_comparison.json').items():add(f'**Temperature {t}:**\n\n```text\n'+ '\n'.join(samples)+'\n```\n')
add('In the starter run, all four samples at 0.8 and 1.2 are identical: increasing temperature does not guarantee a visible change in a small sample. At 0.3 the second sentence uses investment rather than deposit. In the expanded run, the second sample shifts between travel/train, security/system, and data/system as temperature changes. These are observations of four seeded samples per setting, not a statistical quality comparison.\n')
add('## Working chat interface\n\nThe unchanged [chat.py](chat.py) loads the trained expanded `model.pt` and saved vocabulary. [Transcript JSON](evidence/chat_transcript.json), [terminal text](evidence/chat_terminal.txt), [HTML recording player](evidence/chat_recording.html), and [raw terminal recording](evidence/chat_session.cast) document three actual interactions. Download/open the HTML locally to replay it; GitHub displays its source rather than running it. The recording is playback, not a live model interface.\n')
t=read('evidence/chat_transcript.json');add(f"Run: `{runs['Expanded']}`. Model SHA-256 identity: `{t['model_sha256']}`. Prompts were entered automatically into the actual terminal interface, and responses came from nanoGPT.\n")
for turn in t['turns']:add(f"**Prompt:** `{turn['prompt']}`\n\n**Actual response:** {turn['response'] or '[empty response]'}\n\nUnknown prompt tokens: {', '.join(turn['unknown_prompt_words']) or 'none'}.\n")
add('''Each prompt starts fresh; there is no conversation memory. The interface uses a 48-token context, reports truncation/unknown words, uses temperature 0.8, and generates up to 24 tokens. Chat never retrains the model or writes into the corpus. The first two examples fit familiar patterns, while the quantum-computing question produces an unrelated classroom fragment with all prompt tokens unknown.

## Reproduce and launch

From the project folder, create an isolated environment. `requirements-tested.txt` records the exact package versions used on the reported Mac; `requirements-local.txt` provides the teacher's dependency ranges plus the missing NumPy dependency and notebook execution tools for other platforms.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt
```

Windows users can substitute `.venv\\Scripts\\python.exe` for `.venv/bin/python`. For an interactive Jupyter interface, install `requirements-local.txt` as well; VS Code can select the same environment as its notebook kernel.

### Run a fresh experiment without mixing corpora

```sh
.venv/bin/python reproduce.py starter
.venv/bin/python reproduce.py expanded
```

[reproduce.py](reproduce.py) creates a new directory under `reproductions/`, copies the fixed helpers and tests, and runs a fresh notebook there. The starter gets an empty corpus folder; the expanded run gets the two teaching files. This preserves the original executed evidence. Do not simply rerun the starter notebook beside the now-populated corpus folder: classroom mode includes that folder's files.

To prepare files for interactive Jupyter/VS Code without training automatically:

```sh
.venv/bin/python reproduce.py starter --prepare-only
.venv/bin/python reproduce.py expanded --prepare-only
```

Open the newly printed `custom_llm.ipynb` path and run all cells with the Python environment above. Source notebooks and saved evidence remain unchanged. For Colab, upload the prepared notebook and companion helpers/tests, and only the expanded teaching files for the expanded run; save both the executed notebook and results ZIP before ending the temporary session.

### Rerun the four evaluations on saved weights

Choose fresh output directories if preserving previous rerun evidence.

```sh
.venv/bin/python run_evals.py --model llm_runs/20260922T000429_567209Z/model_untrained.pt --stage untrained --output results/starter-untrained-check
.venv/bin/python run_evals.py --model llm_runs/20260922T000429_567209Z/model.pt --output results/starter-final-check
.venv/bin/python run_evals.py --model llm_runs/20260922T001305_992994Z/model_untrained.pt --stage untrained --output results/expanded-untrained-check
.venv/bin/python run_evals.py --model llm_runs/20260922T001305_992994Z/model.pt --output results/expanded-final-check
```

### Start the real chat interface

```sh
.venv/bin/python chat.py --model llm_runs/20260922T001305_992994Z/model.pt --transcript evidence/my_new_chat.json
```

Enter text at `You:` and type `/quit` to exit. Use a new transcript filename each time. The complete saved model is included in the linked run directory and ZIP; `checkpoint.json` only serves the embedding viewer. Neither file is an exact training-resume checkpoint.
''')
for label,r in runs.items():add(f'- [{label} full results ZIP]({r}.zip) · [trained weights]({r}/model.pt) · [untrained weights]({r}/model_untrained.pt).')
add('''
## Limitation and proposed next experiment

The expanded model still fails two inverse spatial questions even though their words are known. A next experiment could introduce varied multi-clause spatial descriptions, balanced across directions, using new objects and wording rather than test stories. Keep the training budget fixed and report coverage, all-case/scorable scores, and unrestricted continuations separately. Improvement is uncertain. Additional untouched tests would be needed to claim unseen generalization.

## Attribution and submission status

Based on the [instructor's sample project](https://github.com/pepealonso95/custom-llm), source commit `9e04ddb6aacb8efcb790e70c62550ca55e0f2a75`. The pinned nanoGPT model comes from Karpathy's commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`; see [nanoGPT license](NANOGPT_LICENSE). [Original starter README](TEACHER_README.md) is preserved. The local contribution consists of configured executed notebooks, original synthetic teaching text, evidence, reproduction helpers, and this analysis.

The local evidence is assembled; student reflection and public publication are still pending. No claim is made that a public repository has been created or checked signed out. Before submission, answer the learning checkpoints, publish the selected evidence in your own public repository, verify notebooks/plots/downloads without signing in, and submit that repository URL through the course portal. The benchmark percentage is not the assignment grade.
''')
(ROOT/'README.md').write_text('\n'.join(L)+'\n')
print('README assembled from saved evidence.')
