# Advanced NLP (SCIA / ANLP1 & ANLP2)

![Banner](static/github_anlp_banner.png)

## Sessions

1. Recap on Deep Learning & basic NLP ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course1_recap.pdf) / [lab session](https://colab.research.google.com/drive/1_QzQBdP289benS8Uo3yPQmtXoM-f80-n?usp=sharing))
2. Tokenization ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course2_tokenization.pdf) / [lab session](https://colab.research.google.com/drive/1xEKz_1LcnkfcEenukIGCrk-Nf_5Hb19s?usp=sharing))
3. Language Modeling ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course3_lm.pdf) / [lab session](https://colab.research.google.com/drive/1QmVOWC1oB206PmOBn8j0EF54laSh3BBd?usp=sharing))
4. NLP without 2048 GPUs ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course4_efficiency.pdf) / lab session)
5. Language Models at Inference Time ([slides](https://raw.githubusercontent.com/NathanGodey/AdvancedNLP/main/slides/pdf/course5_inference.pdf) / [lab session](https://colab.research.google.com/drive/13Q1WVHDvmFX4pDQ9pSr0KrggBnPtBSPX?usp=sharing))
6. Handling the Risks of Language Models ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course6_risks.pdf) / [lab session](https://colab.research.google.com/drive/1BSrIa5p-f2UvJEH-Y0ezniJcOoRHltMm?usp=sharing))
7. Advanced NLP tasks ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course7_advanced.pdf) / [lab session](https://colab.research.google.com/drive/1b43gxnijKdGOU9llPEC-x2dd_E89ROk1?usp=sharing))
8. Domain-specific NLP ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course8_specific.pdf) / [lab session](https://colab.research.google.com/drive/1RUwYAl__CjkkWcsw6gAmlpX3OUszWm-9?usp=sharing))
9. Multilingual NLP ([slides](https://github.com/NathanGodey/AdvancedNLP/blob/main/slides/pdf/Course%209%20-%20Multilingual%20NLP.pdf) / [lab session](https://colab.research.google.com/drive/11TX-q-hAdFiSeMVqFp1VCXhi_Ifoj8Rp?usp=sharing))
10. Multimodal NLP ([slides](https://github.com/NathanGodey/AdvancedNLP/blob/main/slides/pdf/cours10_Multimodal_NLP.pdf) / [lab session](https://colab.research.google.com/drive/1uAA0T6o88QVHeNItS5RxQMrhOV5Lsql4?usp=sharing)

## Evaluation

The evaluation consists in a team project (3-5 people). The choice of the subject is **free** but needs to follow some basic rules:

- Obviously, the project must be highly related with NLP and especially with the notions we will cover in the course
- You can only use open-source LLM that _you serve yourself_. In other words, no API / ChatGPT-like must be used, except for final comparison with your model.
- You must identify and address a <ins>challenging</ins> problem (e.g. not only _can a LLM do X?_, but _can a LLM <ins>that runs on a CPU</ins> do X?_, or _can I make a LLM <ins>better</ins> at X?_)
- It must be reasonably doable: you will not be able to fine-tune (even to use) a 405B parameters model, or to train a model from scratch. That's fine, there are a lot of smaller models that should be good enough, like [the Pythia models](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1), [TinyLLama](https://huggingface.co/collections/TinyLlama/tinyllama-11b-v1-660bb5bfabd8bd25eebbb1ef), the 1B parameter [OLMo](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778), or the small models from the [Llama3.2 suite](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf).

:alarm_clock: The project follows 3 deadlines:

- **Project announcement (before 25/10/24)**: send an email to `nathan.godey@inria.fr` with cc's `matthieu.futeral@inria.fr` and `francis.kulumba@inria.fr` explaining
  - The team members (also cc'ed)
  - A vague description of the project (it can change later on)
- **Project proposal (25% of final grade, before 15/11/24)**: following [this template](https://docs.google.com/document/d/1rCWr6p5N0ip7fpNv9e5wjX7gez4oaFGioatYXRRKGR8/edit?usp=sharing), produce a project proposal explaining first attempts (e.g. version alpha), how they failed/succeeded and what you want to do before the delivery.
- **Project delivery (75% of final grade, 13/12/24)**: delivery of a GitHub repo with an explanatory README + oral presentation on **December 13th**

## Inspiring articles

### Tokenization

- A Vocabulary-Free Multilingual Neural Tokenizer for End-to-End Task Learning (https://arxiv.org/abs/2204.10815)
- BPE-Dropout: Simple and Effective Subword Regularization (https://aclanthology.org/2020.acl-main.170/)
- FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models (https://aclanthology.org/2023.emnlp-main.829/)

### Fast inference

- Efficient Streaming Language Models with Attention Sinks (https://arxiv.org/abs/2309.17453)
- Lookahead decoding (https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
- Efficient Memory Management for Large Language Model Serving with PagedAttention (https://arxiv.org/pdf/2309.06180.pdf)

### Inference-time scaling (OpenAI's o1 model)

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (https://arxiv.org/abs/2201.11903)
- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (https://arxiv.org/abs/2408.03314v1)

### LLM detection

- Detecting Pretraining Data from Large Language Models (https://arxiv.org/abs/2310.16789)
- Proving Test Set Contamination in Black Box Language Models (https://arxiv.org/abs/2310.17623)

### SSMs (off-program)

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (https://arxiv.org/abs/2312.00752)

### Alignment & Safety

- Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (https://aclanthology.org/2020.acl-main.647/)
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (https://arxiv.org/abs/2305.18290)
- Text Embeddings Reveal (Almost) As Much As Text (https://arxiv.org/abs/2310.06816)
