# Two additional wording explorations

All 20 outputs are from the saved expanded nanoGPT. No training, vocabulary edits, corpus changes or fixed-eval changes were made. Temperature 0.8, 24-token maximum, fresh context, and paired seeds 2026–2030 were held fixed. All prompts use known words; none were truncated.

[Protocol before generation](protocol.json) · [All results JSON](results.json) · [CSV](results.csv) · [Unchanged-model audit](audit.json) · [Runner](../../explore_more_phrasings.py)

## teacher_state

A: `today the teacher is`

B: `the teacher today is`

|Seed|A continuation|B continuation|
|---|---|---|
|2026|walking to the bank now .|walking to the bank now .|
|2027|walking to the store now .|walking to the store now .|
|2028|walking to the hospital now .|walking to the hospital now .|
|2029|walking to the market now .|walking to the market now .|
|2030|walking to the bank now .|walking to the bank at the hospital .|

## object_location

A: `a pencil is inside the`

B: `inside the basket is a`

|Seed|A continuation|B continuation|
|---|---|---|
|2026|station today .|basket .|
|2027|hospital now .|window now .|
|2028|box .|box .|
|2029|station today .|box .|
|2030|school today .|box .|

## Observations and limits

The teacher pair has the same meaning and completion slot. Four of five paired continuations match exactly. Seed 2030 differs: A ends with `walking to the bank now .`, while B ends with `walking to the bank at the hospital .`. The second is more awkward and introduces another location, but no factual target was specified; this is a qualitative observation, not a formally scored error.

The location pair is a broader structure exploration, **not a controlled paraphrase comparison**. A leaves a location to complete; B leaves an object to complete. Different output is expected, so paired equality cannot measure success. This design difference was recorded in the protocol before generation.

For A, the model often supplies buildings (station, hospital, school), and once box. A pencil being inside a building is possible; those responses should not automatically be called incorrect. For B, the model supplies basket, window or box. A window inside a basket is unusual in an ordinary scene, while a box or smaller basket inside a basket can be possible. There is no ground-truth object in these prompts. These examples show that grammatical form alone does not establish meaningful spatial reasoning.

Compared with the first exploration, output stability is not universal: the original walking pair matched 5/5 and this teacher pair matched 4/5. These are descriptive counts for two specific pairs, not estimated general paraphrase accuracy. No claim is made that temperature or the model improved.

## A simple reflection addition to review

I tried changing the word order while keeping the model the same. Most of the teacher sentences stayed the same, but one response added another location and sounded less natural. In the location examples, some answers looked unusual, but I could not call every different answer wrong. I learned that I need to look at the full sentence and make sure my comparison is fair.

The additional experiments now contain 45 total continuations: 25 in the first exploration and 20 here. The fixed 48-case benchmark remains separate and unchanged.
