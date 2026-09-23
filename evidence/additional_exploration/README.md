# Additional exploration: wording and temperature

This is an inference-only exploration of the existing expanded model, not another training experiment or a change to the fixed 48-case benchmark. All 25 actual responses are retained.

[Protocol recorded before generation](protocol.json) · [All JSON results](results.json) · [CSV](results.csv) · [Summary](summary.json) · [Unchanged-weights/source audit](audit.json) · [Runner](../../explore_generation.py)

## Controls

Expanded run: 20260922T001305_992994Z; 3,000 training steps; base training learning rate 0.0015. Fresh context for every generation, maximum 24 new tokens, paired seeds 2026–2030. There were no unknown prompt words, no truncated prompts, and no empty responses. The model weights, teaching files and fixed evaluation suite hashes were unchanged. No generated output was added to training.

Prediction: changing word order might change continuation; increasing temperature might increase variation, but neither effect was guaranteed. These are five draws per condition, not a new accuracy benchmark.

## 1. Change wording, keep temperature 0.8

The two prompts describe the same past walking event and end at the destination slot: `last monday she walked to the` and `she walked last monday to the`. Only the placement of the time phrase changes.

|Seed|Original word order: continuation|Reordered: continuation|
|---|---|---|
|2026|station .|station .|
|2027|hospital .|hospital .|
|2028|store .|store .|
|2029|store .|store .|
|2030|school .|school .|

**Finding:** all five paired responses were identical. All continuations supplied plausible destinations. For this specific pair the sampled output was stable under the word-order change. This does not prove that the probability distributions were identical or that arbitrary paraphrases will work. Both prompts share the ending `to the`, so success could reflect a local continuation pattern; it is not evidence that the model fully understood the whole event.

## 2. Change temperature, keep prompt `the customer`

|Seed|Temperature 0.3|Temperature 0.8|Temperature 1.2|
|---|---|---|---|
|2026|recommended the package after checking the price .|compared the package after checking the price .|was quiet yesterday .|
|2027|ordered the offering after checking the price .|ordered the offering after checking the price .|ordered the offering after checking the price .|
|2028|recommended the merchandise after checking the price .|recommended the merchandise after checking the price .|recommended the merchandise after checking the price .|
|2029|selected the offering after checking the price .|ordered the offering after checking the price .|ordered the offering after checking the price .|
|2030|selected the product after checking the price .|selected the product after checking the price .|selected the product after checking the price .|

|Temperature|Distinct full continuations / 5|Distinct generated token types|
|---|---|---|
|0.3|5/5|12|
|0.8|4/5|13|
|1.2|4/5|14|

**Findings:**

- At seed 2026 the continuations changed from `recommended the package after checking the price .` (0.3), to `compared the package after checking the price .` (0.8), to `was quiet yesterday .` (1.2). The last response shifts from a shopping template to a grammar/state template; it is still coherent, not an observed error.
- Seeds 2027, 2028 and 2030 produced the same response at all three temperatures. A setting change need not change every sampled output.
- Distinct full sentences were 5, 4 and 4, not a monotonic increase. Distinct generated token types were 12, 13 and 14. These are different diversity measurements; neither measures correctness. With just five samples, do not conclude that lower temperature is generally more diverse.
- No obvious incoherence appeared in these 15 temperature responses. Higher temperature can raise the chance of less-likely tokens, but this small experiment does not demonstrate increased errors.

## What this adds to the learning discussion

The model can sometimes keep its continuation stable when wording changes. However, this test used a shared ending and cannot establish broad understanding. Temperature changes sampling without changing learned weights. More temperature does not guarantee more distinct sentences in a small sample, and a different continuation is not automatically a worse one. These observations should be reported even though they are less dramatic than the initial prediction.

## Reproduce

From the project folder, run `.venv/bin/python explore_generation.py`. The script intentionally refuses to overwrite an existing output folder. To run again while preserving this evidence, copy the script and change its OUT directory to a new folder outside corpus/. No retraining is needed.
