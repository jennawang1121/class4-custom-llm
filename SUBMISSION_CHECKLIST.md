# Class 4 rubric and submission audit

Audited against the instructor's full assignment text supplied on September 22, 2026. This is an evidence check, **not a promised score**. A low model score is not itself a deduction; explanation quality and valid methodology still require instructor judgment.

## Deliverable quality — 4 possible points

| Requirement | Status | Where the evidence is |
|---|---|---|
| Two executed experiments, not just the template | Complete | [Starter](starter_3000steps.ipynb), [expanded](expanded_3000steps.ipynb): each has 12 executed code cells, no error outputs, all saved outputs, and an added results explanation |
| Clear README grading entry point | Complete | [README](README.md) includes experiment overview, controls, results, interpretation and evidence links |
| Three choices and reasons; pre-training predictions | Complete | Each notebook's original prediction cell; README choices section; 3,000 updates, learning rate 0.0015, classroom/expanded corpus |
| Corpus sources, permissions, extraction review | Complete | [Sources](CORPUS_SOURCES.md), two UTF-8 teaching files, original generator; no PDFs or extraction warnings |
| At least two extension categories and reasons | Complete | Grammar + spatial relations; 576 + 445 unique passages address vocabulary/pattern gaps without reproducing test stories |
| Corpus size, split, vocabulary, both UNK rates | Complete | README and both manifests/vocabulary reports; 4,592 vs 5,613 unique passages, 90/10 split; 136 vs 232 tokens; both corpus UNK rates 0% |
| Actual steps, timing, hardware, parameters | Complete | README and training_summary/config: 3,000 each, CPU/Mac/Python/PyTorch, original loop timings and 111,872/118,016 parameters |
| Untrained/halfway/final samples, no hidden failures | Complete | All 4 samples at each of 0/1,500/3,000 steps in each run, linked and reproduced in README |
| Both loss curves and every measured loss | Complete | Embedded SVG plots, full three-row tables for each run; fixed panels explicitly identified as 20 training + 20 validation passages |
| Token → ID → 64-number vector, before/after | Complete | README displays all 64 initial/final coordinates for customer; both runs' complete inspection/tokenization files linked |
| Real gradient and parameter update | Complete | Both runs' saved first coordinate, before/gradient/actual learning rate/after values; clipping, warmup and AdamW explained |
| Next-token probabilities and learning explanation | Complete | Actual reviewed/compared/customer probability comparison, loss/backpropagation/updates, network weights and causal attention explanation |
| Three temperatures, fixed starting token and seed | Complete | Both complete saved temperature comparisons; BOS and seed 2026; no weight updates; unchanged samples acknowledged |
| Limitation and proposed next experiment | Complete | Spatial failures, unsupported generalization, multi-clause spatial teaching proposal |
| Student reflection | Complete, AI assistance disclosed | [Reviewed simple-English reflection](REFLECTION.md), based on the learner discussion and actual examples |

## Testing and evaluation — 3 possible points

| Requirement | Status | Where the evidence is |
|---|---|---|
| Fixed 48 cases, choices, key, scoring unchanged | Complete | [Suite](evals/language_evals.json), [runner](run_evals.py); original file hashes verified |
| All four full result sets | Complete | Starter and expanded × untrained/final: 192 unique case records across four CSV/JSON/summary sets |
| Four-row comparison in README | Complete | 9/48 → 21/48 and 5/48 → 28/48; all four result links present |
| All-case score, scorable accuracy, coverage | Complete | Three distinct metrics, correct denominators, missing coverage explained |
| All groups/categories and free continuations | Complete | Full group/category tables; all case outputs; examples where correct choice coexists with nonsensical continuation |
| Explain vocabulary versus pattern learning | Complete | Six newly scorable cases; four learned successes; two remaining spatial failures; cross-run confounds explicitly acknowledged |
| Corpus/test/vocabulary separation | Complete | [Independent provenance audit](evidence/corpus_separation_audit.json): actual corpus/splits/vocabulary reconstructed; 160 reserved passages excluded; no added evaluation artifacts |
| No source-file/unseen-generalization claims | Complete | Passage split and shared templates disclosed; public development benchmark label |
| Rerunnable saved-model evaluation | Complete | All four checkpoint reruns exactly reproduce per-case results in [verification](evidence/verification.json) |
| Extra exploration reported honestly | Complete | 45 additional continuations; controls, small-sample limitations, and the non-equivalent location completion slots disclosed; not substituted for the 48-case suite |

## Working result — 3 possible points

| Requirement | Status | Where the evidence is |
|---|---|---|
| Supplied nanoGPT, real model weights | Complete | Pinned model source/hash, PyTorch, original architecture, actual model.pt and model_untrained.pt for both experiments |
| Rerunnable training instructions | Complete | [reproduce.py](reproduce.py) keeps starter corpus empty and expanded corpus separate; both fresh reruns matched original losses, inspections and scores |
| Working chat interface and source | Complete | Original [chat.py](chat.py), exact launch command; actual model and saved vocabulary loaded |
| Three actual interactions including limitation | Complete | [Chat evidence](evidence/README.md): familiar sentence, grammar continuation and unknown-topic failure; model identity verified |
| Screenshot or recording | Complete | Actual terminal [asciicast recording](evidence/chat_session.cast), [HTML replay](evidence/chat_recording.html), and [transcript](evidence/chat_transcript.json). Download/open HTML locally; GitHub does not execute it inline |
| Tiny-model label, context, unknown words, fresh prompts | Complete | Interface prints limits and warnings; README explains 48-token context and no conversation memory or chat retraining |
| Public repository and downloadable evidence | Verified for core artifacts | [Repository](https://github.com/jennawang1121/class4-custom-llm); [public checks](evidence/public_access_check.json) |

## Final submission checks and remaining caveats

- [x] Both executed notebooks and both complete results ZIPs preserved locally and publicly.
- [x] Model downloads, four complete result sets, plots, reflection and chat source/evidence included.
- [x] Public README and loss plots rendered while signed out.
- [x] Added GitHub-readable [starter](evidence/notebook_exports/starter.md) and [expanded](evidence/notebook_exports/expanded.md) notebook exports containing every cell and saved output as backups.
- [x] **Native GitHub notebook preview:** both executed notebooks rendered while signed out. The 3,000-step completion, final loss, final evaluation, vector outputs and one SVG figure in each were verified. Initial loading delays resolved.
- [ ] **Local embedding-viewer walkthrough:** all numeric initial/final vectors and cosine neighbors were checked, but the automated browser blocked the local file URL. Open embedding-viewer.html yourself and load a saved checkpoint.json; compare customer before/after. Interactive viewer use is not claimed verified.
- [ ] **Course portal submission:** paste the public repository URL into the assignment submission portal. Publication is not the same as portal submission.

No further model optimization is required. The remaining issues concern viewing and submission, not a need to raise the model score. The instructor may still judge the clarity and depth of the explanation; this checklist cannot guarantee 10/10.
