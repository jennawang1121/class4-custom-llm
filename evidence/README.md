# Actual chat evidence

Model: `llm_runs/20260922T001305_992994Z/model.pt`

Model identity: `c731332d0636cf076a4f366457b5b3313bb20453b9ae0334bdf8b8e47e2de2a4`

Three prompts were entered automatically into the actual interactive terminal interface using a pseudo-terminal. All responses were produced by the saved nanoGPT, not by an external model. The terminal stream was recorded as asciicast v2; the HTML player replays that stream without network access.

[Replay recording](chat_recording.html) · [Raw recording](chat_session.cast) · [JSON transcript](chat_transcript.json) · [Terminal text](chat_terminal.txt)

## Launch from the project folder

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt
.venv/bin/python chat.py --model llm_runs/20260922T001305_992994Z/model.pt --transcript evidence/my_new_chat.json
```

Choose a new transcript filename each time. Type /quit to finish. The model file must exist at the specified path; it is saved by expanded_3000steps.ipynb. The complete expanded results ZIP also contains model.pt. checkpoint.json is not the inference model.

The interface uses a 48-token context and reports unknown words or truncation. Each prompt starts fresh, temperature is 0.8, and at most 24 new tokens are generated. Chat does not train the model or enter corpus/.

## Interactions

**Prompt:** the customer

**Actual reply:** compared the package after checking the price .

Unknown words: none

**Prompt:** last monday she

**Actual reply:** walked to the store .

Unknown words: none

**Prompt:** Can you explain quantum computing?

**Actual reply:** store has a question about the different shopper and the local buyer and the different shopper and the support with another subscriber and the

Unknown words: ?, can, computing, explain, quantum, you

## Interpretation

The first two replies fit familiar classroom/grammar patterns. The third fails to answer the question and instead generates a classroom-style fragment; all six prompt tokens are unknown. It is a narrow sentence-continuation model, not a general question-answering assistant. These three demonstrations are not a new accuracy benchmark.
