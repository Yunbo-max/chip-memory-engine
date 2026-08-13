# LLaMA Contribution-Flow Trust Case

Verdict: `PASS`

- Examples: 100
- Model: /root/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee
- Signal: 71120-style contribution-flow + retrieval relevance
- Alpha contribution-flow weight: 0.65
- Trust threshold: 0.650
- Baseline F1: 0.7181
- Contribution-flow F1: 0.7102
- F1 delta: -0.0079
- Steps: 4.0000 -> 2.8300
- Effort reduction: 29.25%
- Answer preservation rate: 96.00%
- Peak CUDA allocation: 13102.6 MiB

This uses the real 71120 LLaMA contribution-matrix code path, but with the locally available LLaMA-2-7B chat checkpoint rather than the exact LLaMA-3.2/Gemma paper checkpoints.
