# AdvancedNLP

## Sessions
1. Recap on Deep Learning & basic NLP ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course1_recap.pdf) / [lab session](https://colab.research.google.com/drive/1_QzQBdP289benS8Uo3yPQmtXoM-f80-n?usp=sharing))
2. Tokenization ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course2_tokenization.pdf) / [lab session](https://colab.research.google.com/drive/1xEKz_1LcnkfcEenukIGCrk-Nf_5Hb19s?usp=sharing))
3. Language Modeling ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course3_lm.pdf) / [lab session](https://colab.research.google.com/drive/1QmVOWC1oB206PmOBn8j0EF54laSh3BBd?usp=sharing))
4. NLP without 2048 GPUs ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course4_efficiency.pdf) / [lab session](https://colab.research.google.com/drive/12OZwC5t8nUrh6JUNkG76XDMTDk19jP2m?usp=sharing))
5. Handling the Risks of Language Models ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course5_risks.pdf) / [lab session](https://drive.google.com/file/d/1m46UbyhAejGt6KKMoitxtC5rXD8dhrZr/view?usp=sharing))
6. Advanced NLP tasks ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course6_advanced.pdf) / [lab session](https://colab.research.google.com/drive/1Owh2KH6dPkJIkz0Bsi5XbOnZN6uAhAF2?usp=sharing))
7. Domain-specific NLP ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course7_specific.pdf) / [lab session])
8. Multilingual NLP ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/Course%205%20-%20Multilingual%20NLP.pdf) / [lab session](https://colab.research.google.com/drive/11TX-q-hAdFiSeMVqFp1VCXhi_Ifoj8Rp?usp=sharing))
9. Multimodal NLP ([slides](https://docs.google.com/presentation/d/1K2DgnPSOGXB1hQ4FZoUU-5ppJ4dn_sLC41Ecwmxi2Zk/edit?usp=sharing) / [lab session](https://colab.research.google.com/drive/1uAA0T6o88QVHeNItS5RxQMrhOV5Lsql4?usp=sharing))

## Evaluation
⚠️ **There is no oral presentation/evaluation for this course.**

The evaluation consists in a team project (3-5 people). There are two options:
- **Demo** : Use a well-known approach to produce a MVP for an <ins>original</ins> use-case and present it in a demo.
  - *Example: An online platform that detects AI-generated text.*<br>
- **R&D** : Based on a research article, conduct original experiments and produce a report. (see [Potential articles](#potential-articles))
  - *Example: Do we need Next Sentence Prediction in BERT? (Answer: No)*

It will consist of three steps:
- **Team announcement (before 15/12/23)**: send an email to `nathan.godey@inria.fr` with cc's `matthieu.futeral@inria.fr` and `francis.kulumba@inria.fr` explaining
  - The team members (also cc'ed)
  - Type of project and vague description (can change afterwards)
- **Project plan (30% of final grade, before 07/01/23)**: following [this template](https://docs.google.com/document/d/1rCWr6p5N0ip7fpNv9e5wjX7gez4oaFGioatYXRRKGR8/edit?usp=sharing), produce a project plan explaining first attempts (e.g. version alpha), how they failed/succeeded and what you want to do before the delivery.
- **Project delivery (70% of final grade, before mid-February)**: deliver a `nb_team_members * 2` pages project report and a GitHub repo (more details coming soon)

## Potential articles
### Tokenization
- A Vocabulary-Free Multilingual Neural Tokenizer for End-to-End Task Learning (https://arxiv.org/abs/2204.10815)
- BPE-Dropout: Simple and Effective Subword Regularization (https://aclanthology.org/2020.acl-main.170/)

### Fast inference
- Efficient Streaming Language Models with Attention Sinks (https://arxiv.org/abs/2309.17453)
- Lookahead decoding (https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
- Efficient Memory Management for Large Language Model Serving with PagedAttention (https://arxiv.org/pdf/2309.06180.pdf)

### LLM detection
- Detecting Pretraining Data from Large Language Models (https://arxiv.org/abs/2310.16789)
- Proving Test Set Contamination in Black Box Language Models (https://arxiv.org/abs/2310.17623)

### SSMs (off-program)
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (https://arxiv.org/abs/2312.00752)

### Alignment & Safety
- Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (https://aclanthology.org/2020.acl-main.647/)
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (https://arxiv.org/abs/2305.18290)
- Text Embeddings Reveal (Almost) As Much As Text (https://arxiv.org/abs/2310.06816)

