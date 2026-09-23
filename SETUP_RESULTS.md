# Setup run: 10 steps, learning rate 0.0015

Completed classroom-only pipeline check; not the final experiment.
Run: `llm_runs/20260921T224253_008282Z`.
Executed notebook: [setup_10steps.ipynb](setup_10steps.ipynb).

| Step | Training loss | Validation loss |
|---|---|---|
|0|4.926252841949463|4.927548408508301|
|5|4.234650611877441|4.220412731170654|
|10|4.066014766693115|4.077120304107666|

Fixed panels contain 20 training and 20 validation documents.
All 48 evals were processed at both stages: correct 9/48 before, 8/48 after.
Scorable accuracy: 9/24 (37.5%) before, 8/24 (33.33%) after. Coverage stayed 50%.
All 24 extension cases were unscorable with the starter vocabulary.
Loss decreased, but four-choice accuracy did not improve. This does not establish whether 0.0015 is better than 0.001: no controlled comparison was run.
The generated chat reply remains an incoherent word sequence.

Original tests and model source were preserved. NumPy was added to local dependencies because the evaluation runner requires it for hashing. An earlier attempt stopped before training because NumPy was missing; its incomplete run folder is not final evidence. The notebook was then rerun from the beginning successfully.

Formal training budget, extension categories, and the learner explanation checkpoints remain pending.
