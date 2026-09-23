# Two-experiment evidence report

Both experiments trained fresh nanoGPT models for 3,000 steps at base learning rate 0.0015 with seed 42. The expanded experiment added 576 grammar and 445 spatial teaching passages. [Sources and design](CORPUS_SOURCES.md). No PDF sources or extraction warnings.

These are public development tests. The test file and scoring are unchanged; no test prompts, keys, or outputs were added to teaching inputs. Vocabulary comes only from training passages.

|Experiment / stage|All-case success|Scorable accuracy|Coverage|Evidence|
|---|---|---|---|---|
|Starter untrained|9/48 (18.75%)|9/24 (37.50%)|50.00%|[JSON](llm_runs/20260922T000429_567209Z/language_evals/untrained/eval_results.json), [CSV](llm_runs/20260922T000429_567209Z/language_evals/untrained/eval_results.csv)|
|Starter final|21/48 (43.75%)|21/24 (87.50%)|50.00%|[JSON](llm_runs/20260922T000429_567209Z/language_evals/final/eval_results.json), [CSV](llm_runs/20260922T000429_567209Z/language_evals/final/eval_results.csv)|
|Expanded untrained|5/48 (10.42%)|5/30 (16.67%)|62.50%|[JSON](llm_runs/20260922T001305_992994Z/language_evals/untrained/eval_results.json), [CSV](llm_runs/20260922T001305_992994Z/language_evals/untrained/eval_results.csv)|
|Expanded final|28/48 (58.33%)|28/30 (93.33%)|62.50%|[JSON](llm_runs/20260922T001305_992994Z/language_evals/final/eval_results.json), [CSV](llm_runs/20260922T001305_992994Z/language_evals/final/eval_results.csv)|

## Interpretation

Expanded final grammar is 3/3 and spatial relations 1/3. All six are newly scorable, but coverage alone is insufficient: two spatial cases remain wrong. The other 18 extension cases remain unscorable. The original 24 cases improve from 21/24 to 24/24, but changed vocabulary, initialization dimensions and data distribution prevent attributing this to a single cause. A shared seed does not yield identical weights when vocabulary dimensions change.

A notable failure: lang_41 selects inside instead of below. Lang_42 selects north instead of right. These expose limitations in transferring single-sentence teaching to multi-sentence inverse-relation prompts. This is an interpretation, not a proven causal diagnosis.

Four-choice success can hide poor free text: lang_26 correctly selects are, but its free continuation is "traffic discussion of course now ." Lang_40 selects book, yet its free continuation is only "." All results are retained.

A next experiment could add varied multi-clause relational exercises with distinct stories, balancing directions and holding the public suite fixed. Unseen generalization would require additional untouched tests.

## Starter

[Executed notebook](starter_3000steps.ipynb) · [Complete results ZIP](llm_runs/20260922T000429_567209Z.zip)

Vocabulary: 136; parameters: 111872; train/validation passages: 4132/460; training/validation unknown-token rates: 0.0/0.0. Both rates are corpus measurements, distinct from test coverage.

### Full fixed-panel loss table

Each panel contains 20 documents. Panels remain fixed within a run; added data changes panels across runs. Losses across vocabularies are not directly comparable.

|Step|Train|Validation|
|---|---|---|
|0|4.926252841949463|4.927548408508301|
|1500|0.6793705224990845|0.7186149954795837|
|3000|0.6775152683258057|0.7051523923873901|

![Training curves](llm_runs/20260922T000429_567209Z/training_curves.svg)

### All category scores

|Category|Before correct / scorable / total|After correct / scorable / total|
|---|---|---|
|categories_and_analogies|0 / 0 / 3|0 / 0 / 3|
|domain_context|3 / 8 / 8|8 / 8 / 8|
|domain_place|3 / 8 / 8|8 / 8 / 8|
|everyday_knowledge|0 / 0 / 3|0 / 0 / 3|
|grammar|0 / 0 / 3|0 / 0 / 3|
|negation|0 / 0 / 3|0 / 0 / 3|
|new_wording|3 / 8 / 8|5 / 8 / 8|
|opposites|0 / 0 / 3|0 / 0 / 3|
|reference|0 / 0 / 3|0 / 0 / 3|
|sequence|0 / 0 / 3|0 / 0 / 3|
|spatial_relations|0 / 0 / 3|0 / 0 / 3|

### Complete sample timeline

[step_0000](llm_runs/20260922T000429_567209Z/samples/step_0000.txt)

```text
pear professor bond doctor course harvest team physician journey checking buyer delivery traffic report the lecturer item offering and system <UNK> taste recommended mentioned bus question customer at mortgage nurse in instructor
kitchen purchase journey product question discussion journey service . nurse local
compared and purchase update mortgage question loan taste in market treatment learned another item bicycle product bicycle focused data and dentist recommended mango apple taxi bicycle delivery peach quality update student lesson
important hospital juice patient return recommended deposit tutor returned understand kitchen student design ordered hospital treatment important package traffic with yesterday investment important of mentioned store ordered mortgage nurse shopper the station
```

[step_1500](llm_runs/20260922T000429_567209Z/samples/step_1500.txt)

```text
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
we learned about the important website during a discussion of data .
our school has a question about the different instructor and course .
```

[step_3000](llm_runs/20260922T000429_567209Z/samples/step_3000.txt)

```text
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
the report about the nurse explains the health in detail .
the consumer compared the offering after checking the price .
```

### Inspections and reproducibility

- [config.json](llm_runs/20260922T000429_567209Z/config.json)
- [corpus_manifest.json](llm_runs/20260922T000429_567209Z/corpus_manifest.json)
- [vocabulary_report.json](llm_runs/20260922T000429_567209Z/vocabulary_report.json)
- [split.json](llm_runs/20260922T000429_567209Z/split.json)
- [tokenization.json](llm_runs/20260922T000429_567209Z/tokenization.json)
- [inspection.json](llm_runs/20260922T000429_567209Z/inspection.json)
- [training.csv](llm_runs/20260922T000429_567209Z/training.csv)
- [training_summary.json](llm_runs/20260922T000429_567209Z/training_summary.json)
- [temperature_comparison.json](llm_runs/20260922T000429_567209Z/temperature_comparison.json)
- [eval_separation.json](llm_runs/20260922T000429_567209Z/eval_separation.json)
## Expanded

[Executed notebook](expanded_3000steps.ipynb) · [Complete results ZIP](llm_runs/20260922T001305_992994Z.zip)

Vocabulary: 232; parameters: 118016; train/validation passages: 5051/562; training/validation unknown-token rates: 0.0/0.0. Both rates are corpus measurements, distinct from test coverage.

### Full fixed-panel loss table

Each panel contains 20 documents. Panels remain fixed within a run; added data changes panels across runs. Losses across vocabularies are not directly comparable.

|Step|Train|Validation|
|---|---|---|
|0|5.4747138023376465|5.467024326324463|
|1500|0.7467778325080872|0.7647263407707214|
|3000|0.7317943572998047|0.7450159192085266|

![Training curves](llm_runs/20260922T001305_992994Z/training_curves.svg)

### All category scores

|Category|Before correct / scorable / total|After correct / scorable / total|
|---|---|---|
|categories_and_analogies|0 / 0 / 3|0 / 0 / 3|
|domain_context|2 / 8 / 8|8 / 8 / 8|
|domain_place|2 / 8 / 8|8 / 8 / 8|
|everyday_knowledge|0 / 0 / 3|0 / 0 / 3|
|grammar|0 / 3 / 3|3 / 3 / 3|
|negation|0 / 0 / 3|0 / 0 / 3|
|new_wording|1 / 8 / 8|8 / 8 / 8|
|opposites|0 / 0 / 3|0 / 0 / 3|
|reference|0 / 0 / 3|0 / 0 / 3|
|sequence|0 / 0 / 3|0 / 0 / 3|
|spatial_relations|0 / 3 / 3|1 / 3 / 3|

### Complete sample timeline

[step_0000](llm_runs/20260922T001305_992994Z/samples/step_0000.txt)

```text
harvest design learned book lecturer update orange update review review order bag south during cabinet the child educator in <BOS> today map merchandise explains today toward delivery book tutor book report beside
looking platform walks update left played was farmer harvest child package talked service garden up every walks course product left was they cabinet picture dog below fruit drawer purchase juice small coin
rested offering below shelf plant walked looking nurse map new teachers were learning product map window of walks instructor consumer garden delivery report explains course farmer client car orange i mortgage product
design program children system professor surgeon now system it north application toward update another chair from sits chair drivers he ruth surgeon last security platform patient juice at are apple reviewed therapist
```

[step_1500](llm_runs/20260922T001305_992994Z/samples/step_1500.txt)

```text
today the bank focused on payment and the local credit .
today the office focused on data and the local system .
today the kitchen focused on juice and the important orange .
today the market focused on delivery and the important merchandise .
```

[step_3000](llm_runs/20260922T001305_992994Z/samples/step_3000.txt)

```text
today the bank focused on payment and the local investment .
today the office focused on security and the local system .
today the hospital focused on patient and the important doctor .
the important subscriber was mentioned in the service report yesterday .
```

### Inspections and reproducibility

- [config.json](llm_runs/20260922T001305_992994Z/config.json)
- [corpus_manifest.json](llm_runs/20260922T001305_992994Z/corpus_manifest.json)
- [vocabulary_report.json](llm_runs/20260922T001305_992994Z/vocabulary_report.json)
- [split.json](llm_runs/20260922T001305_992994Z/split.json)
- [tokenization.json](llm_runs/20260922T001305_992994Z/tokenization.json)
- [inspection.json](llm_runs/20260922T001305_992994Z/inspection.json)
- [training.csv](llm_runs/20260922T001305_992994Z/training.csv)
- [training_summary.json](llm_runs/20260922T001305_992994Z/training_summary.json)
- [temperature_comparison.json](llm_runs/20260922T001305_992994Z/temperature_comparison.json)
- [eval_separation.json](llm_runs/20260922T001305_992994Z/eval_separation.json)

## Saved-model check

The expanded model was loaded from model.pt and the unchanged runner reproduced 28/48. [Rerun summary](results/expanded_saved_model_check/eval_summary.json).

```sh
.venv/bin/python run_evals.py --model llm_runs/20260922T001305_992994Z/model.pt --output results/new-eval-check
```

The submission README now contains the learning explanations; evidence/ contains three actual chat interactions and a terminal recording. Student reflection and public publication remain pending.
