# Class 4: training and evaluating a tiny nanoGPT

Two fresh nanoGPT models were trained on CPU for **3,000 steps at learning rate 0.0015**: first on the supplied classroom corpus, then on the classroom corpus plus original grammar and spatial-relation teaching text. The expanded model scored **28/48** on the unchanged public development tests, versus **21/48** for the trained starter. It still failed two spatial questions and could not score 18 other extension questions because of missing vocabulary. It is a narrow sentence-completion model, not a general chatbot.

This report was prepared with AI assistance from actual notebook outputs. The student selected the settings and extension categories. The [reflection](REFLECTION.md) uses simple English based on the student’s questions, explanations, and actual experiment results. It was revised with AI assistance and reviewed in the learning conversation.

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
|---|---|---|---|---|---|
|Starter|untrained|9/48 (18.75%)|24/48 (50.00%)|37.50%|[CSV](llm_runs/20260922T000429_567209Z/language_evals/untrained/eval_results.csv) · [JSON](llm_runs/20260922T000429_567209Z/language_evals/untrained/eval_results.json) · [summary](llm_runs/20260922T000429_567209Z/language_evals/untrained/eval_summary.json)|
|Starter|final|21/48 (43.75%)|24/48 (50.00%)|87.50%|[CSV](llm_runs/20260922T000429_567209Z/language_evals/final/eval_results.csv) · [JSON](llm_runs/20260922T000429_567209Z/language_evals/final/eval_results.json) · [summary](llm_runs/20260922T000429_567209Z/language_evals/final/eval_summary.json)|
|Expanded|untrained|5/48 (10.42%)|30/48 (62.50%)|16.67%|[CSV](llm_runs/20260922T001305_992994Z/language_evals/untrained/eval_results.csv) · [JSON](llm_runs/20260922T001305_992994Z/language_evals/untrained/eval_results.json) · [summary](llm_runs/20260922T001305_992994Z/language_evals/untrained/eval_summary.json)|
|Expanded|final|28/48 (58.33%)|30/48 (62.50%)|93.33%|[CSV](llm_runs/20260922T001305_992994Z/language_evals/final/eval_results.csv) · [JSON](llm_runs/20260922T001305_992994Z/language_evals/final/eval_results.json) · [summary](llm_runs/20260922T001305_992994Z/language_evals/final/eval_summary.json)|

### Group and category results

Entries are **correct / scorable / total**. Zero scorable cases means missing coverage, not a measured 0% accuracy among scorable cases.

|Group|Starter before|Starter after|Expanded before|Expanded after|
|---|---|---|---|---|
|extend_corpus|0 / 0 / 24|0 / 0 / 24|0 / 6 / 24|4 / 6 / 24|
|starter_patterns|6 / 16 / 16|16 / 16 / 16|4 / 16 / 16|16 / 16 / 16|
|starter_transfer|3 / 8 / 8|5 / 8 / 8|1 / 8 / 8|8 / 8 / 8|

|Category|Starter before|Starter after|Expanded before|Expanded after|
|---|---|---|---|---|
|categories_and_analogies|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|
|domain_context|3 / 8 / 8|8 / 8 / 8|2 / 8 / 8|8 / 8 / 8|
|domain_place|3 / 8 / 8|8 / 8 / 8|2 / 8 / 8|8 / 8 / 8|
|everyday_knowledge|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|
|grammar|0 / 0 / 3|0 / 0 / 3|0 / 3 / 3|3 / 3 / 3|
|negation|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|
|new_wording|3 / 8 / 8|5 / 8 / 8|1 / 8 / 8|8 / 8 / 8|
|opposites|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|
|reference|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|
|sequence|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|0 / 0 / 3|
|spatial_relations|0 / 0 / 3|0 / 0 / 3|0 / 3 / 3|1 / 3 / 3|

The trained-model gain of seven correct cases comprises three additional starter-transfer successes plus four newly covered extension successes. The six new scorable cases are the three grammar and three spatial questions. Their expanded untrained score was 0/6, and trained score was 4/6; this is consistent with learning beyond vocabulary inclusion, but does not isolate a causal mechanism. Vocabulary size and initialization also changed between experiments.

Concrete limitations in the expanded run:

- `lang_41` selected `inside` rather than `below`; `lang_42` selected `north` rather than `right`. Knowing the words did not guarantee inverse spatial reasoning.
- `lang_26` correctly ranked `are` among four choices, but freely continued with `traffic discussion of course now .`.
- `lang_40` ranked `book` highest among its four choices, but its free continuation was only `.`. Its probability for `book` was only about 0.0057% of the whole vocabulary. Being best among four candidates does not make a word likely overall.
- The other 18 extension cases remain unscorable. No claim is made that this model has mastered those skills.

The full case records linked above preserve prompts, choice probabilities, status, selected answers, and actual continuations, including failures.

### Separation from training

A fresh [corpus separation audit](evidence/corpus_separation_audit.json) reconstructed the saved training text, train/validation splits, and vocabulary from the permitted teaching sources. It found no added test prompts, answer-key material, or evaluation outputs, and confirmed the submission ZIP contains the same teaching files.

The suite stayed in `evals/`; only teaching files were placed in `corpus/`. Test prompts, paired reference answers, scoring rules, evaluation outputs, and chat transcripts were never used to build the vocabulary or update weights. Each run excluded 160 generated classroom passages containing reserved test prefixes before splitting/vocabulary construction. Imported files passed the supplied normalized-prefix check. The suite's original file SHA-256 remains `e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7` (the runner also reports a separately computed canonical suite hash). Exact matching is not a semantic leakage detector; the added teaching text was also reviewed for copied test stories and answer lists.

- Starter: [separation check](llm_runs/20260922T000429_567209Z/eval_separation.json) · [file manifest](llm_runs/20260922T000429_567209Z/corpus_manifest.json) · [training text](llm_runs/20260922T000429_567209Z/corpus.txt).
- Expanded: [separation check](llm_runs/20260922T001305_992994Z/eval_separation.json) · [file manifest](llm_runs/20260922T001305_992994Z/corpus_manifest.json) · [training text](llm_runs/20260922T001305_992994Z/corpus.txt).

## Run settings, data, and complete loss history

The 90/10 split is by deduplicated passage, not source file. Training and held-out text share templates, so held-out loss tests combinations within this restricted distribution. It does not establish transfer to new domains or unseen templates. Vocabulary is built only from the training split, retaining at most 509 ordinary types plus UNK/BOS/EOS; neither formal run reached that limit.

|Run|Completed steps|Training-loop seconds|Parameters|Vocabulary|Train / validation passages|Train / held-out UNK rate|
|---|---|---|---|---|---|---|
|Starter|3000|9.285453|111,872|136|4132 / 460|0.0% / 0.0%|
|Expanded|3000|9.456151|118,016|232|5051 / 562|0.0% / 0.0%|

Both used CPU on macOS 15.7.7 arm64, Python 3.12.14 and PyTorch 2.14.0. Timings are the original notebook training-loop measurements, not download/setup or total notebook wall time. Neither formal run was interrupted. An initial setup attempt failed before training because NumPy was absent; NumPy was installed and the notebook rerun from the beginning. The successful 10-step setup is [preserved separately](setup_10steps.ipynb); the incomplete attempt is excluded from submission evidence.

### Starter loss

Each fixed panel has **20 documents** (the notebook supports at most 20). Loss is averaged over non-padding next-token targets, not all corpus text. Every measured value is below.

|Step|Training loss|Validation loss|
|---|---|---|
|0|4.926252841949463|4.927548408508301|
|1500|0.6793705224990845|0.7186149954795837|
|3000|0.6775152683258057|0.7051523923873901|

![Starter loss](llm_runs/20260922T000429_567209Z/training_curves.svg)

[history.json](llm_runs/20260922T000429_567209Z/history.json) · [training.csv](llm_runs/20260922T000429_567209Z/training.csv) · [config](llm_runs/20260922T000429_567209Z/config.json) · [training summary](llm_runs/20260922T000429_567209Z/training_summary.json) · [vocabulary report](llm_runs/20260922T000429_567209Z/vocabulary_report.json)

### Expanded loss

Each fixed panel has **20 documents** (the notebook supports at most 20). Loss is averaged over non-padding next-token targets, not all corpus text. Every measured value is below.

|Step|Training loss|Validation loss|
|---|---|---|
|0|5.4747138023376465|5.467024326324463|
|1500|0.7467778325080872|0.7647263407707214|
|3000|0.7317943572998047|0.7450159192085266|

![Expanded loss](llm_runs/20260922T001305_992994Z/training_curves.svg)

[history.json](llm_runs/20260922T001305_992994Z/history.json) · [training.csv](llm_runs/20260922T001305_992994Z/training.csv) · [config](llm_runs/20260922T001305_992994Z/config.json) · [training summary](llm_runs/20260922T001305_992994Z/training_summary.json) · [vocabulary report](llm_runs/20260922T001305_992994Z/vocabulary_report.json)

## Untrained, halfway, and final samples

Every saved sample is included below, including garbled text. Each stage uses BOS, sampling seed 2026, temperature 0.8, four samples, and a 32-token limit. This sample timeline differs from the 24-token evaluation continuations.

### Starter: step 0

[Complete sample file](llm_runs/20260922T000429_567209Z/samples/step_0000.txt)

```text
pear professor bond doctor course harvest team physician journey checking buyer delivery traffic report the lecturer item offering and system <UNK> taste recommended mentioned bus question customer at mortgage nurse in instructor
kitchen purchase journey product question discussion journey service . nurse local
compared and purchase update mortgage question loan taste in market treatment learned another item bicycle product bicycle focused data and dentist recommended mango apple taxi bicycle delivery peach quality update student lesson
important hospital juice patient return recommended deposit tutor returned understand kitchen student design ordered hospital treatment important package traffic with yesterday investment important of mentioned store ordered mortgage nurse shopper the station
```

### Starter: step 1500

[Complete sample file](llm_runs/20260922T000429_567209Z/samples/step_1500.txt)

```text
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
we learned about the important website during a discussion of data .
our school has a question about the different instructor and course .
```

### Starter: step 3000

[Complete sample file](llm_runs/20260922T000429_567209Z/samples/step_3000.txt)

```text
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
the report about the nurse explains the health in detail .
the consumer compared the offering after checking the price .
```

### Expanded: step 0

[Complete sample file](llm_runs/20260922T001305_992994Z/samples/step_0000.txt)

```text
harvest design learned book lecturer update orange update review review order bag south during cabinet the child educator in <BOS> today map merchandise explains today toward delivery book tutor book report beside
looking platform walks update left played was farmer harvest child package talked service garden up every walks course product left was they cabinet picture dog below fruit drawer purchase juice small coin
rested offering below shelf plant walked looking nurse map new teachers were learning product map window of walks instructor consumer garden delivery report explains course farmer client car orange i mortgage product
design program children system professor surgeon now system it north application toward update another chair from sits chair drivers he ruth surgeon last security platform patient juice at are apple reviewed therapist
```

### Expanded: step 1500

[Complete sample file](llm_runs/20260922T001305_992994Z/samples/step_1500.txt)

```text
today the bank focused on payment and the local credit .
today the office focused on data and the local system .
today the kitchen focused on juice and the important orange .
today the market focused on delivery and the important merchandise .
```

### Expanded: step 3000

[Complete sample file](llm_runs/20260922T001305_992994Z/samples/step_3000.txt)

```text
today the bank focused on payment and the local investment .
today the office focused on security and the local system .
today the hospital focused on patient and the important doctor .
the important subscriber was mentioned in the service report yesterday .
```

The untrained samples are mostly disconnected words. Halfway and final samples largely use the supplied classroom templates. Some samples are unchanged across stages; the complete timeline above preserves this. Plausible template text is narrower evidence than broad language understanding.

## How learning works, using recorded values

A **corpus** is the teaching text. A **token** is the unit processed by this notebook: a whole word or punctuation mark after normalization. A token ID is an arbitrary lookup index. A **vector** is an ordered list of numbers; an **embedding** is a trainable vector assigned to a token. Its coordinates are not pre-labeled human concepts.

In the starter run, `customer` maps to **ID 28**, which selects a row of **64 numbers**. In the expanded vocabulary it maps to **ID 50**, illustrating that IDs depend on the vocabulary and are not meanings. Each ID stays fixed within its own run. [Starter tokenization](llm_runs/20260922T000429_567209Z/tokenization.json), [expanded tokenization](llm_runs/20260922T001305_992994Z/tokenization.json).

The starter's complete recorded `customer` vectors are shown below (rounded to nine decimal places; the inspection JSON retains full precision).

**embedding_before (64 coordinates):**

```text
-0.057591915, -0.004809953, 0.042631887, 0.019338956, 0.015643112, -0.028824365, 0.025609056, 0.000052454
0.024706816, 0.020691765, 0.007369017, -0.033089615, -0.053547867, -0.005742930, -0.024166763, -0.014716119
0.004685706, -0.010454268, -0.008381076, -0.018258560, -0.020133700, 0.005098642, -0.010916502, -0.012633352
0.028389625, -0.002631223, -0.004071926, 0.013641920, -0.009891724, -0.016717626, 0.001906079, -0.001453504
0.016026523, -0.005674875, -0.000672348, -0.001290727, -0.007319473, -0.000930707, 0.001507625, -0.004976639
-0.028987018, 0.018092988, -0.007348014, -0.005440254, 0.015641203, -0.004543506, 0.041567937, 0.052355435
0.022642685, -0.015414278, -0.025121203, -0.006797463, 0.029352751, -0.002533680, 0.029801227, -0.022797002
-0.030237909, 0.006436788, 0.050490811, 0.007490998, -0.010722861, 0.024737332, -0.014468740, 0.013235915
```

**embedding_after (64 coordinates):**

```text
0.027679948, -0.029153936, 0.140851974, 0.148320183, 0.061166734, 0.066308074, 0.171408191, 0.049818415
-0.067805350, -0.003911192, 0.052914146, -0.070505589, -0.045772001, -0.014890255, -0.137181520, -0.039631102
-0.182766303, -0.166018531, -0.016975679, -0.038197204, -0.095464863, 0.032952219, -0.078541316, 0.012205401
0.001340303, -0.072483256, 0.127457470, -0.055964880, 0.053609062, -0.153659269, -0.081742890, 0.083740108
-0.056837745, 0.145162806, 0.086735852, -0.024371011, 0.038787279, -0.154155895, 0.118365958, -0.028139861
0.145180732, -0.028214030, -0.134522662, 0.023568243, -0.016124114, -0.092494503, 0.006635461, 0.046019055
0.005363967, -0.179476008, 0.010172208, -0.127456605, -0.005834099, 0.073494978, 0.114161521, 0.052926540
0.080297485, 0.001453290, 0.037701964, 0.075109355, 0.150374547, -0.027490210, 0.088820629, -0.073004447
```

A neural network combines many learned weights: token/position embeddings, attention projections, and feed-forward transformations, with nonlinear GELU operations, normalization, and residual connections. Its final scores become a probability distribution via softmax. Attention mixes information from earlier tokens and the current token; the causal mask blocks future positions so the model cannot peek at the word it must predict. The notebook's recorded first-head attention is one part of the actual computation, not a universal explanation of the model.

For next-token learning, the loss penalizes low probability on the observed next word. PyTorch backpropagation computes **gradients**: how the current loss would locally change if a parameter changed slightly. AdamW uses them, its running statistics, weight decay, and the learning-rate schedule to update weights. The embedding is one set of these trainable parameters.

### One real gradient and first update

|Run / customer coordinate 0|Before|Gradient (before clipping)|Actual first-step learning rate|After|Change|
|---|---|---|---|---|---|
|Starter|-0.057591915130615234|0.0006925869965925813|1.5e-05|-0.05760690197348595|-1.49868428707e-05|
|Expanded|0.0056657949462533|-0.002263699658215046|1.5e-05|0.005680793896317482|1.49989500642e-05|

Warmup makes the first update use **0.000015**, although the configured base rate is 0.0015. In the starter example the recorded gradient is positive and this coordinate decreases slightly. That is one actual update, distinct from the much larger net change after 3,000 steps. The gradient is recorded before gradient clipping; the code clips the overall norm before AdamW updates. These values should not be interpreted as a simple SGD calculation.

### A next-token probability change

For the same prefix `the customer`, in the starter run:

|Candidate|Before training|After training|
|---|---|---|
|reviewed|0.711145%|17.718221%|
|compared|0.620728%|15.728493%|
|customer|1.600694%|0.008334%|

These changes come from the whole trained network, not just one embedding coordinate. Generation samples a token from the distribution, appends it to the context, and repeats until EOS or the output limit. A rising correct-word probability can reduce loss without changing which of four candidates ranks first; this helps explain why loss and benchmark accuracy need not move together.

- [Starter: full vectors, gradients, probabilities, and attention](llm_runs/20260922T000429_567209Z/inspection.json) · [viewer checkpoint](llm_runs/20260922T000429_567209Z/checkpoint.json).
- [Expanded: full vectors, gradients, probabilities, and attention](llm_runs/20260922T001305_992994Z/inspection.json) · [viewer checkpoint](llm_runs/20260922T001305_992994Z/checkpoint.json).

### Embedding neighbors

Cosine similarity below uses all 64 dimensions of the starter embeddings, excludes the token itself, and was computed from the saved checkpoint. It is not a test score.

- Before: bus (0.213), educator (0.203), helped (0.202), bank (0.201), risk (0.198).
- After: shopper (0.983), buyer (0.980), subscriber (0.974), consumer (0.971), client (0.968).

The neighbors reflect the restricted contexts in the teaching sentences, not human-defined semantic labels. To inspect visually, open [embedding-viewer.html](embedding-viewer.html) locally and load the desired checkpoint.json. PCA compresses 64 dimensions for display and can distort apparent distances; cosine neighbors use the full vectors. The viewer is not a chat interface.

## Temperature: inference without retraining

Temperature divides output scores before softmax. Lower temperature concentrates probability on favored tokens; higher temperature flattens it. It changes sampling, not vocabulary or learned weights. The notebook resets sampling seed 2026 and starts at BOS for each temperature; all saved samples follow.

### Starter

[Full temperature comparison](llm_runs/20260922T000429_567209Z/temperature_comparison.json)

**Temperature 0.3:**

```text
our school has a question about the new educator and lesson .
a review of risk helped us understand the different investment .
the report about the nurse explains the health in detail .
the local consumer was mentioned in the purchase report yesterday .
```

**Temperature 0.8:**

```text
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
the report about the nurse explains the health in detail .
the consumer compared the offering after checking the price .
```

**Temperature 1.2:**

```text
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
the report about the nurse explains the health in detail .
the consumer compared the offering after checking the price .
```

### Expanded

[Full temperature comparison](llm_runs/20260922T001305_992994Z/temperature_comparison.json)

**Temperature 0.3:**

```text
today the bank focused on risk and the local investment .
a review of travel helped us understand the local train .
today the hospital focused on patient and the important doctor .
the important subscriber was mentioned in the service report yesterday .
```

**Temperature 0.8:**

```text
today the bank focused on payment and the local investment .
today the office focused on security and the local system .
today the hospital focused on patient and the important doctor .
the important subscriber was mentioned in the service report yesterday .
```

**Temperature 1.2:**

```text
today the bank focused on payment and the local investment .
today the office focused on data and the local system .
today the hospital focused on patient and the important doctor .
today the market focused on delivery and the important merchandise .
```

In the starter run, all four samples at 0.8 and 1.2 are identical: increasing temperature does not guarantee a visible change in a small sample. At 0.3 the second sentence uses investment rather than deposit. In the expanded run, the second sample shifts between travel/train, security/system, and data/system as temperature changes. These are observations of four seeded samples per setting, not a statistical quality comparison.

## Working chat interface

The unchanged [chat.py](chat.py) loads the trained expanded `model.pt` and saved vocabulary. [Transcript JSON](evidence/chat_transcript.json), [terminal text](evidence/chat_terminal.txt), [HTML recording player](evidence/chat_recording.html), and [raw terminal recording](evidence/chat_session.cast) document three actual interactions. Download/open the HTML locally to replay it; GitHub displays its source rather than running it. The recording is playback, not a live model interface.

Run: `llm_runs/20260922T001305_992994Z`. Model SHA-256 identity: `c731332d0636cf076a4f366457b5b3313bb20453b9ae0334bdf8b8e47e2de2a4`. Prompts were entered automatically into the actual terminal interface, and responses came from nanoGPT.

**Prompt:** `the customer`

**Actual response:** compared the package after checking the price .

Unknown prompt tokens: none.

**Prompt:** `last monday she`

**Actual response:** walked to the store .

Unknown prompt tokens: none.

**Prompt:** `Can you explain quantum computing?`

**Actual response:** store has a question about the different shopper and the local buyer and the different shopper and the support with another subscriber and the

Unknown prompt tokens: ?, can, computing, explain, quantum, you.

Each prompt starts fresh; there is no conversation memory. The interface uses a 48-token context, reports truncation/unknown words, uses temperature 0.8, and generates up to 24 tokens. Chat never retrains the model or writes into the corpus. The first two examples fit familiar patterns, while the quantum-computing question produces an unrelated classroom fragment with all prompt tokens unknown.

## Reproduce and launch

From the project folder, create an isolated environment. `requirements-tested.txt` records the exact package versions used on the reported Mac; `requirements-local.txt` provides the teacher's dependency ranges plus the missing NumPy dependency and notebook execution tools for other platforms.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt
```

Windows users can substitute `.venv\Scripts\python.exe` for `.venv/bin/python`. For an interactive Jupyter interface, install `requirements-local.txt` as well; VS Code can select the same environment as its notebook kernel.

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

- [Starter full results ZIP](llm_runs/20260922T000429_567209Z.zip) · [trained weights](llm_runs/20260922T000429_567209Z/model.pt) · [untrained weights](llm_runs/20260922T000429_567209Z/model_untrained.pt).
- [Expanded full results ZIP](llm_runs/20260922T001305_992994Z.zip) · [trained weights](llm_runs/20260922T001305_992994Z/model.pt) · [untrained weights](llm_runs/20260922T001305_992994Z/model_untrained.pt).

## Limitation and proposed next experiment

The expanded model still fails two inverse spatial questions even though their words are known. A next experiment could introduce varied multi-clause spatial descriptions, balanced across directions, using new objects and wording rather than test stories. Keep the training budget fixed and report coverage, all-case/scorable scores, and unrestricted continuations separately. Improvement is uncertain. Additional untouched tests would be needed to claim unseen generalization.

## Additional inference-only exploration

We also compared two word orders at temperature 0.8 and temperatures 0.3/0.8/1.2 for `the customer`, using the same five seeds per condition. All 25 continuations are preserved in the [additional exploration report](evidence/additional_exploration/README.md).

The two word orders produced identical paired responses in all five cases, though both end with `to the`, so this is only a narrow stability observation. At seed 2026, raising temperature changed the shopping continuation to `was quiet yesterday .`, still a coherent sentence. Distinct full responses numbered 5, 4 and 4 across temperatures: small samples need not show increasing diversity. No weights, vocabulary, teaching text, or fixed test cases changed.

A [second wording exploration](evidence/additional_phrasings/README.md) adds 20 responses. The teacher word-order pair matched in 4/5 paired samples. The location prompts changed the completion slot, so they are reported as a broader qualitative exploration rather than a controlled paraphrase test. All 45 additional continuations are retained; no extra benchmark accuracy is claimed.

## Attribution and submission status

Based on the [instructor's sample project](https://github.com/pepealonso95/custom-llm), source commit `9e04ddb6aacb8efcb790e70c62550ca55e0f2a75`. The pinned nanoGPT model comes from Karpathy's commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`; see [nanoGPT license](NANOGPT_LICENSE). [Original starter README](TEACHER_README.md) is preserved. The local contribution consists of configured executed notebooks, original synthetic teaching text, evidence, reproduction helpers, and this analysis.

The experimental evidence and reviewed reflection are assembled. Before submission, verify notebooks, plots and downloads in the public repository without signing in, then submit the repository URL through the course portal. The benchmark percentage is not the assignment grade.

