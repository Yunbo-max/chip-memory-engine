# LLaMA Contribution-Flow Trust Case

Verdict: `PASS`

- Examples: 50
- Model: /root/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee
- Signal: 71120-style contribution-flow + retrieval relevance
- Alpha contribution-flow weight: 0.65
- Trust threshold: 0.650
- Baseline F1: 0.8183
- Contribution-flow F1: 0.8297
- F1 delta: 0.0113
- Steps: 3.9400 -> 2.6000
- Effort reduction: 34.01%
- Answer preservation rate: 94.00%
- Peak CUDA allocation: 13103.6 MiB

This uses the real 71120 LLaMA contribution-matrix code path, but with the locally available LLaMA-2-7B chat checkpoint rather than the exact LLaMA-3.2/Gemma paper checkpoints.
