# AdvancedNLP

## Sessions
1. Recap on Deep Learning & basic NLP ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course1_recap.pdf) / [lab session](https://colab.research.google.com/drive/1_QzQBdP289benS8Uo3yPQmtXoM-f80-n?usp=sharing))
2. Tokenization ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course2_tokenization.pdf) / [lab session](https://colab.research.google.com/drive/1xEKz_1LcnkfcEenukIGCrk-Nf_5Hb19s?usp=sharing))
3. Language Modeling ([slides](https://github.com/NathanGodey/AdvancedNLP/raw/main/slides/pdf/course3_lm.pdf) / [lab session](https://colab.research.google.com/drive/1QmVOWC1oB206PmOBn8j0EF54laSh3BBd?usp=sharing))
4. Handling the Risks of Language Models ([slides] / [lab session])
5. NLP without 2048 GPUs ([slides] / [lab session])
6. Advanced NLP tasks ([slides] / [lab session])
7. Domain-specific NLP ([slides] / [lab session])
8. Multilingual NLP ([slides] / [lab session])
9. Mutlimodal NLP ([slides] / [lab session])

## Evaluation
The evaluation consists in a team project (3-5 people). There are two options:
- **Demo** : Use a well-known approach to produce a MVP for an <ins>original</ins> use-case and present it in a demo.
  - *Example: An online platform that detects AI-generated text.*<br>
- **R&D** : Based on a research article, conduct original experiments and produce a report. (see [Potential articles](#potential-articles))
  - *Example: Do we need Next Sentence Prediction in BERT? (Answer: No)*

It will consist of three steps:
- **Team announcement (before 15/12/23)**: send an email to `nathan.godey@inria.fr` with cc's `matthieu.futeral@inria.fr` and `francis.kulumba@inria.fr` explaining
  - The team members (also cc'ed)
  - Type of project and vague description (can change afterwards)
- **Project plan (before 07/01/23)**: following [this template](https://docs.google.com/document/d/1rCWr6p5N0ip7fpNv9e5wjX7gez4oaFGioatYXRRKGR8/edit?usp=sharing), produce a project plan explaining first attempts (e.g. version alpha), how they failed/succeeded and what you want to do before the delivery.
- **Project delivery (before mid-February)**: deliver a `nb_team_members * 2` pages project report and a GitHub repo (more details coming soon)
 
## Potential articles
### Tokenization
- A Vocabulary-Free Multilingual Neural Tokenizer for End-to-End Task Learning (https://arxiv.org/abs/2204.10815)
- BPE-Dropout: Simple and Effective Subword Regularization (https://aclanthology.org/2020.acl-main.170/)

### Fast inference
- Efficient Streaming Language Models with Attention Sinks (https://arxiv.org/abs/2309.17453)

### LLM detection
- Detecting Pretraining Data from Large Language Models (https://arxiv.org/abs/2310.16789)
- Proving Test Set Contamination in Black Box Language Models (https://arxiv.org/abs/2310.17623)

### Alignment & Safety
- Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (https://aclanthology.org/2020.acl-main.647/)
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (https://arxiv.org/abs/2305.18290)
- Text Embeddings Reveal (Almost) As Much As Text (https://arxiv.org/abs/2310.06816)
