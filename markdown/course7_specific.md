---
theme: gaia
_class: lead
paginate: true
title: "Course 7: Domain-Specific NLP"
backgroundColor: #fff
marp: true
---


# **Domain-Specific NLP**

---

<!--footer: "Course 6: Domain-Specific NLP" -->


### Contents

1. Domain-Specific Models
    a. _Don’t Stop Pre-training_
    b. Specialized Models (BioBERT, SciBERT, Galactica)
2. Unsupervised Classification Models
    a. Out-of-the-box representations: limitations
    b. SimCSE, E5, GTE...
    c. Document Representation: DocBERT
3. Learning Long-Range Dependencies
    a. Long-range attention models
    b. State-space models: S4

---



<!--_class: lead -->
## Domain-Specific Models

---


### Domain-Specific Models

**Pretrained (Large) Language Models** are trained on content crawled over the internet, books, reports and news papers and are, hence **are open-domain**.

A **textual domain** is the **distribution over language characterizing a given topic or genre** [1].

* You are more likely to see the word "integer" in computer science than in news papers.
* An (L)LM will be more perplex to the word "integer" even though the input comes from a StackOverflow post.

---


### Don’t Stop Pretraining

_Don't Stop Pretraining: Adapt Language Models to Domains and Tasks._ [1]

![height:350px](https://d3i71xaburhd42.cloudfront.net/e816f788767eec6a8ef0ea9eddd0e902435d4271/1-Figure1-1.png)

---


### Don’t Stop Pretraining

![width:1100px](https://d3i71xaburhd42.cloudfront.net/e816f788767eec6a8ef0ea9eddd0e902435d4271/3-Table1-1.png)

<sub><sup>**Table 1**: List of the domain-specific unlabeled datasets. In columns 5 and 6, we report ROBERTA’s masked LM loss on 50K randomly sampled held-out documents from each domain before $(\mathcal{L}_{ROB.})$ and after $(\mathcal{L}_{DAPT})$ $DAPT$ (lower implies a better fit on the sample). ‡ indicates that the masked LM loss is estimated on data sampled from sources similar to ROBERTA’s pretraining corpus.</sup></sub>

---


### Don’t Stop Pretraining

![height:300px](https://d3i71xaburhd42.cloudfront.net/e816f788767eec6a8ef0ea9eddd0e902435d4271/3-Figure2-1.png)

<sub><sup>**Figure 2**: Vocabulary overlap (%) between domains. PT denotes a sample from sources similar to ROBERTA’s pretraining corpus. Vocabularies for each domain are created by considering the top 10K most frequent words (excluding stopwords) in documents sampled from each domain.</sup></sub>

---


### Don’t Stop Pretraining

![width:800px](https://d3i71xaburhd42.cloudfront.net/e816f788767eec6a8ef0ea9eddd0e902435d4271/6-Table5-1.png)

<sub><sup>**Table 5**: Results on different phases of adaptive pretraining compared to the baseline RoBERTa (col. 1). Our approaches are $DAPT$ (col. 2, §3), $TAPT$ (col. 3, §4), and a combination of both (col. 4).</sup></sub>

---


### Don’t Stop Pretraining

"We show that **pretraining the model towards a specific task or small corpus can provide significant benefits**. Our findings suggest it may be valuable to complement work on ever-larger LMs with parallel efforts to **identify and use domain and task relevant corpora to specialize models**."

---


### BioBERT

"[..] the word distributions of general and biomedical corpora are quite different, which can often be a problem for biomedical text mining models." [2]

---


### BioBERT

![width:900px](https://d3i71xaburhd42.cloudfront.net/1e43c7084bdcb6b3102afaf301cce10faead2702/3-Table1-1.png)

<sub><sup>**Table 1**. List of text corpora used for BioBERT</sup></sub>

---


### BioBERT

"We showed that **pre-training BERT on biomedical corpora is crucial in applying it to the biomedical domain**. Requiring minimal task-specific architectural modification, **BioBERT outperforms previous models on biomedical text mining tasks** such as NER, RE and QA."

---


### SciBERT

"[...] while both BERT and ELMo have released pretrained models, they are still trained on general domain corpora such as news articles and Wikipedia." [3]

---


### SciBERT

![width:800px](https://d3i71xaburhd42.cloudfront.net/156d217b0a911af97fa1b5a71dc909ccef7a8028/4-Table1-1.png)

<sub><sup>**Table 1**: Test performances of all BERT variants on all tasks and datasets. [...]</sup></sub>

---


### SciBERT

![width:900px](https://d3i71xaburhd42.cloudfront.net/156d217b0a911af97fa1b5a71dc909ccef7a8028/4-Table2-1.png)

<sub><sup>**Table 2**: Comparing SciBERT with the reported BioBERT results on biomedical datasets. </sup></sub>

---


### SciBERT

NB: SciBERT was trained on curated textual data ; not trained on code or script for example --at leat not trained directly and purposefully on this kind of data

---


### Galactica

"Computing has indeed revolutionized how research is conducted, but information overload remains an overwhelming problem [...]. In this paper, we argue for a better way through large language models. Unlike search engines, language models can potentially store, combine and reason about scientific knowledge." [4]

* Galactica was trained on a rather small highly curated dataset.
* All the data was standardized as markdown text.

---


### Galactica

![width:750px](https://d3i71xaburhd42.cloudfront.net/7d645a3fd276918374fd9483fd675c28e46506d1/4-Table1-1.png)

<sub><sup>**Table 1**: Tokenizing Nature. Galactica trains on text sequences that represent scientific phenomena.  </sup></sub>

---


### Galactica

1. **Citations:** we wrap citations with special reference tokens [START_REF] and [END_REF].
2. **Step-by-Step Reasoning:** we wrap step-by-step reasoning with a working memory token <work>, mimicking an internal working memory context.
3. **Mathematics:** for mathematical content, with or without LaTeX, we split ASCII operations into individual characters. Parentheses are treated like digits. The rest of the operations allow for unsplit repetitions. Operation characters are !"#$%&’*+,-./:;<=>?\^_‘| and parentheses are ()[]{}.

---


4. **Numbers:** we split digits into individual tokens. For example 737612.62 -> 7,3,7,6,1,2,.,6,2.
5. **SMILES formula:** we wrap sequences with [START_SMILES] and [END_SMILES] and apply characterbased tokenization. Similarly we use [START_I_SMILES] and [END_I_SMILES] where isomeric SMILES is denoted. For example, C(C(=O)O)N → C,(,C,(,=,O,),O,),N.
6. **Amino acid sequences:** we wrap sequences with [START_AMINO] and [END_AMINO] and apply character-based tokenization, treating each amino acid character as a single token. For example, MIRLGAPQTL -> M,I,R,L,G,A,P,Q,T,L.

---


7. **DNA sequences:** we also apply a character-based tokenization, treating each nucleotide base as a token, where the start tokens are [START_DNA] and [END_DNA]. For example, CGGTACCCTC -> C, G, G, T, A, C, C, C, T, C.

---

### Galactica

![width:750px](https://d3i71xaburhd42.cloudfront.net/7d645a3fd276918374fd9483fd675c28e46506d1/9-Figure5-1.png)

<sub><sup>**Figure 5:** Prompt Pre-training. Pre-training weighs all tokens equally as part of the self-supervised loss. This leads to a weak relative signal for tasks of interest, meaning model scale has to be large to work. Instruction tuning boosts performance post hoc, and can generalize to unseen tasks of interest, but it risks performance in tasks that are distant from instruction set tasks. Prompt pre-training has a weaker task of interest bias than instruction tuning but less risk of degrading overall task generality.</sup></sub>

---


### Galactica

* **GeLU Activation** - GeLU activations for all model sizes.
* **Context Window** - a 2048 length context window.
* **No Biases** - following PaLM, we do not use biases in any of the dense kernels or layer norms.
* **Learned Positional Embeddings** - learned positional embeddings for the model.
* **Vocabulary** - vocabulary of 50k tokens using BPE. The vocabulary was generated from a randomly selected 2% subset of the training data.

---


### Galactica

_Gaussian Error Linear Units function (GeLu)_

$$GELU(x)= x ∗ \Phi(x)$$

Where $\Phi(x)$ is the Gaussian function.

$$GELU(x) \approx x ∗ \frac{1}{2}(1 + Tanh(\frac{2}{\pi}∗(x+0.044715 ∗ x^{3})))$$

---


### Galactica

![height:350px](https://pytorch.org/docs/stable/_images/GELU.png)

* Allows small negative values when $x < 0$.
* Avoids the dying ReLU problem.

---


### Galactica

_Why no biases?_

![height:400px](https://upload.wikimedia.org/wikipedia/commons/9/91/Full_GPT_architecture.png)

---



<!--_class: lead -->
## Unsupervised Classification Models

---


### Out-of-the-box representations

pass

---



<!--_class: lead -->
## Sentiment Analysis

---


### Sentiment Analysis

**Sentiment analysis** is a sentence classification task aiming at **automatically mapping data to their sentiment**.

It can be **binary** classification (e.g., positive or negative) or **multiclass** (e.g., enthusiasm, anger, etc)

---


### Sentiment Analysis

![width:650px](https://media.geeksforgeeks.org/wp-content/uploads/20230802120409/Single-Sentence-Classification-Task.png)

---


### Sentiment Analysis

The loss can be the likes of cross-entropy (CE), binary cross-entropy (BCE) or KL-Divergence (KL).

$$\mathcal{L}_{CE} = - \frac{1}{N} \sum_{n'=1}^{N}y^{(n)}.log(f(\textbf{x}, \theta)^{(n)})$$

$$\mathcal{L}_{BCE} = - y^{(n)}.log(f(\textbf{x}, \theta)^{(n)}) + (1 - y^{(n)}).(1 - f(\textbf{x}, \theta)^{(n)})$$

$$\mathcal{L}_{KL} = - \frac{1}{N} \sum_{n'=1}^{N}y^{(n)}.log(\frac{y^{(n)}}{f(\textbf{x}, \theta)^{(n)}})$$

---


<!--_class: lead -->
## Question Answering (QA)

---


### Question Answering (QA)

**QA** is the task of **retrieving a span of text from a context** that is best suited to answer a question.

This task is extractive -> **information retrieval**

---


### Question Answering (QA)

![width:1000px](https://miro.medium.com/v2/resize:fit:1093/1*UgytWW_huSrfWtGUV5vmNQ.png)

---


### Question Answering (QA)

![width:1150px](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter7/qa_labels.svg)

---

### Question Answering (QA)

The loss is the cross entropy over the output of the starting token and the ending one:

$$\mathcal{L}_{CE_{QA}} = \mathcal{L}_{CE_{start}} + \mathcal{L}_{CE_{end}}$$

---



<!--_class: lead -->
## Natural Language Inference (NLI)

---


### Natural Language Inference (NLI)

**NLI** is the task of **determining whether a "hypothesis" is true (entailment), false (contradiction), or undetermined (neutral)** given a "premise". [1]

---


### Natural Language Inference (NLI)

Premise|Label|Hypothesis
-------|-----|----------
A man inspects the uniform of a figure in some East Asian country.|contradiction|The man is sleeping.
An older and younger man smiling.|neutral|Two men are smiling and laughing at the cats playing on the floor.
A soccer game with multiple males playing.|entailment|Some men are playing a sport.

---


### Natural Language Inference (NLI)

![width:550px](https://nlp.gluon.ai/_images/bert-sentence-pair.png)

---


### Natural Language Inference (NLI)

The loss is simply the cross entropy or the divergence over the output of the `CLS` token and the true label.

$$\mathcal{L}_{NLI} = \mathcal{L}_{CE_{CLS}}$$

We are trying to compress the information about both sentence in one `CLS` token via attention and decide about their relationship.

Is it possible to help the model infering more information with les text data?

---


### Going Further: LM as Knowledge Graphs

Yasunaga, M., Bosselut, A., Ren, H., Zhang, X., Manning, C. D., Liang, P. S., & Leskovec, J. (2022). [Deep bidirectional language-knowledge graph pretraining](https://arxiv.org/abs/2210.09338). Advances in Neural Information Processing Systems, 35, 37309-37323.

---


### Going Further: LM as Knowledge Graphs

![height:500px](../imgs/course6/dragon_sampling.PNG)

---


### Going Further: LM as Knowledge Graphs

![height:500px](../imgs/course6/dragon_training.PNG)

---


### Going Further: LM as Knowledge Graphs

This architecture ***involves a KG ready to use beforeheaad and pre-training from scratch***. How can we better **perform NLP task without having to retrain or fine-tune** a model?

---



<!--_class: lead -->
## Exploit LLMs capacities: Chain-of-thoughts & In context Learning

---


### Exploit LLMs capacities

ICL enables LLMs to learn new tasks using natural language prompts without explicit retraining or fine-tuning.

The efficacy of ICL is closely tied to the model's scale, training data quality, and domain specificity.

---


### Exploit LLMs capacities

![height:500px](https://thegradient.pub/content/images/size/w800/2023/04/icl-copy2.png)

---


### Exploit LLMs capacities

![height:500px](https://lh6.googleusercontent.com/In6MiddAKdLNEjwHeOzkIJlK3FmZank8f2ibBERPReIwTAKkDm4HglsizdjE8O23gmjyPaEFJSMsdRZLiVx5vNE6RLY2pyukmSEh9acYSwBCUNljXpcalKK4d0KUvcRNlEsNG7x4Exn7jDOEHDwbyE0)

---


### Exploit LLMs capacities

![width:1100px](https://thegradient.pub/content/images/size/w1000/2023/04/Screen-Shot-2023-04-19-at-8.09.07-PM.png)

![width:600px](https://lh6.googleusercontent.com/L_cA-kq0nkDAPO76ju9z8m_3KmZ8nyOIvXrOPoQ9ldAXCR0ACtFOanfCYUllb2g9OBa-2nG5BnsgjKuEPXSlbmgbRNqbS9p3vldqark5wAaTWnGsJofzNzK3GKUsww6byRCgA_AmHcItRgPLoFSk8N0)

---


### Exploit LLMs capacities

![height:500px](https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fzero-cot.79793bee.png&w=1080&q=75)

---



<!--_class: lead -->
## Questions?

---


### References

[1] Gururangan, Suchin, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. “[Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks.](https://doi.org/10.18653/v1/2020.acl-main.740.)” In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, edited by Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, 8342–60. Online: Association for Computational Linguistics, 2020.

---

[2] Lee, Jinhyuk, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. “[BioBERT: A Pre-Trained Biomedical Language Representation Model for Biomedical Text Mining.](https://doi.org/10.1093/bioinformatics/btz682)” Bioinformatics 36, no. 4 (February 15, 2020): 1234–40.

[3] Beltagy, Iz, Kyle Lo, and Arman Cohan. “[SciBERT: A Pretrained Language Model for Scientific Text.](https://doi.org/10.18653/v1/D19-1371)” In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), edited by Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, 3615–20. Hong Kong, China: Association for Computational Linguistics, 2019.

---

[4] Taylor, Ross, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. “[Galactica: A Large Language Model for Science.](https://doi.org/10.48550/arXiv.2211.09085)” arXiv, November 16, 2022.
