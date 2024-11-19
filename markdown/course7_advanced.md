---
theme: gaia
_class: lead
paginate: true
title: "Course 7: Advanced NLP Tasks"
backgroundColor: #fff
marp: true
---


# **Advanced NLP Tasks**

---

<!--footer: "Course 7: Advanced NLP Tasks" -->
<!--_class: lead -->
## Introduciton

---


### Introduction

**Information extraction (IE)** is the task of **automatically extracting structured information from unstructured** and/or **semi-structured** machine-readable **documents** and other electronically represented sources.

---

### Introduction

As NLP evolves, so do IE tasks. Traditional tasks evolve, and new ones emerge out of necessity. 

What are the most common IE tasks, and what are their related tasks?

---


## Contents

1. **Named Entity Recognition (NER)**
    a. Part-of-Speech Tagging (POS)
    b. Conditional Random Field (CRF)
2. **Sentiment Analysis**
3. **QuestionAnswering (QA)**
4. **Natural Language Inference (NLI)**
5. **Going further: LM as knowledge graphs**
6. **Exploit LLMs capacities: Chain-of-thoughts & In context learning**

---


<!--_class: lead -->
## Named Entity Recognition (NER)

---


<!--footer: "Named Entity Recognition (NER)" -->
### Part-of-Speech Tagging (POS)

Named entity recognition (**NER**), aims at **identifying real-world entity mentions from texts**, and **classifying them** into **predefined types**.

Example:
"<span style="color:purple;">Suxamethonium</span> infusion rate and observed <span style="color:blue;">fasciculations</span>."

"<span style="color:purple;">Suxamethonium chloride</Span> (<span style="color:purple;">Sch</span>) was administred i.v."

---


### Part-of-Speech Tagging (POS)

We wish to predict an output vector $\textbf{y} = (y_{1}, y_{1}, ..., y_{L})$, of random variables, given an observed characteristic vector $\textbf{x} = (x_{1}, x_{2}, ..., x_{L})$

$\textbf{y}$ takes it value from a list of $N$ possible values.

---


### Part-of-Speech Tagging (POS)

POS is the process of mapping words in a text with a label corresponding to their grammatical class.

("He", "likes", "to", "drink", "tea"), $\rightarrow$ ("PERSONAL PRONOUN", "VERB", "TO", "VERB", "NOUN").

---


### Part-of-Speech Tagging (POS)

There several levels of granularity.: using [the tag set for english](https://www.ibm.com/docs/en/wca/3.5.0?topic=analytics-part-speech-tag-sets)

("He", "likes", "to", "drink", "tea"), $\rightarrow$ ("PRP", "VBP", "TO", "VB", "NN").

---


### Conditional Random Field (CRF)

<center><img width="600px" src="https://ubiai.tools/wp-content/uploads/2023/12/bert-for-ner.png"/></center>

---


### Conditional Random Field (CRF)

For each token in a sentence at position $l$ we want to compute a probability $p$ to belong to a class $n$.

$$p: f(\textbf{x}, \theta)_{l} \mapsto ?$$
with $p \in [0, 1]$

---


### Conditional Random Field (CRF)

Using the softmax function?

$$p: f(\textbf{x}, \theta)_{l}^ \mapsto \frac{e^{f(\textbf{x}, \theta)_{l}^{(n)}}}{\sum_{n'=1}^{N}e^{f(\textbf{x}, \theta)^{(n')}_{l}}}$$

The probability given by the softmax function will not encode non-local dependencies!

---


### Conditional Random Field (CRF)

We need to take sequential decisions: what if we add transition scores into our softmax?

$$p: f(\textbf{x}, \theta)_{l} \mapsto \frac{e^{f(\textbf{x}, \theta)_{l}^{(n)} + t(y^{(n)}_{l}, y_{l-1})}}{\sum_{n'=1}^{N}e^{f(\textbf{x}, \theta)_{l}^{(n')} + t(y^{(n')}_{l}, y_{l-1})}}$$

But this is the probability for one token to belong to a class, we want to compute the probability of a whole sequence of label at once...

---


### Conditional Random Field (CRF)

$$\begin{flalign}
P(\textbf{y}|\textbf{x}) &= \prod_{l=2}^{L}p(\textbf{y}|f(\textbf{x}, \theta)_{l})\\
\\
&= \prod_{l=2}^{L}\frac{e^{f(\textbf{x}, \theta)_{l}^{(n)} + t(y^{(n)}_{l}, y_{l-1})}}{\sum_{n'=1}^{N}e^{f(\textbf{x}, \theta)_{l}^{(n')} + t(y^{(n')}_{l}, y_{l-1})}}\\
\end{flalign}$$

---

### Conditional Random Field (CRF)

$$\begin{flalign}
P(\textbf{y}|\textbf{x}) &= \frac{exp[{\sum_{l=2}^{L}\textbf{(}f(\textbf{x}, \theta)_{l}^{(n)} + t(y^{(n)}_{l}, y_{l-1})}\textbf{)}]}{\sum_{n'=1}^{N}exp[{\sum_{l=2}^{L}\textbf{(}f(\textbf{x}, \theta)_{l}^{(n')} + t(y^{(n')}_{l}, y_{l-1})}\textbf{)}]}\\
\\
&= \frac{exp[{\sum_{l=2}^{L}\textbf{(}U(\textbf{x}, y^{(n)}_{l}) + T(y^{(n)}_{l}, y_{l-1})}\textbf{)}]}{\sum_{n'=1}^{N}exp[{\sum_{l=2}^{L}\textbf{(}U(\textbf{x}, y^{(n')}_{l}) + T(y^{(n')}_{l}, y_{l-1})}\textbf{)}]}\\
\\
&= \frac{exp[{\sum_{l=2}^{L}\textbf{(}U(\textbf{x}, y^{(n)}_{l}) + T(y^{(n)}_{l}, y_{l-1})}\textbf{)}]}{Z(\textbf{x})}

\end{flalign}$$

---


### Conditional Random Field (CRF)

$Z(\textbf{x})$ is commonly referred as the partition function. However, its not trivial to compute: we'll end up with a complexity of $\mathcal{O}(N^{L})$.

Where $N$ is the number of possible labels and $L$ the sequence length.

How do we proceed?

---


### Conditional Random Field (CRF)

<center><img height="500px" src="https://raw.githubusercontent.com/PythonWorkshop/intro-to-nlp-with-pytorch/master/images/viterbi.png"/></center>

---


### Conditional Random Field (CRF)

<center><img height="500px" src="https://raw.githubusercontent.com/PythonWorkshop/intro-to-nlp-with-pytorch/master/images/crf_transition_matrix.png"/></center>

---


### Conditional Random Field (CRF)

<center><img height="500px" src="https://raw.githubusercontent.com/PythonWorkshop/intro-to-nlp-with-pytorch/master/images/linear_crf_example.png"/></center>

---


### Conditional Random Field (CRF)

Negative log-likelihood:

$$\begin{flalign}
\mathcal{L} &= -log(P(\textbf{y}|\textbf{x}))\\

&= -log(\frac{exp[{\sum_{l=2}^{L}\textbf{(}U(\textbf{x}, y^{(n)}_{l}) + T(y^{(n)}_{l}, y_{l-1})}\textbf{)}]}{Z(\textbf{x})})\\

&= -[log(exp[{\sum_{l=2}^{L}\textbf{(}U(\textbf{x}, y^{(n)}_{l}) + T(y^{(n)}_{l}, y_{l-1})}\textbf{)}]) - log(Z(\textbf{x}))]\\

&= log(Z(\textbf{x})) - {\sum_{l=2}^{L}\textbf{(}U(\textbf{x}, y^{(n)}_{l}) + T(y^{(n)}_{l}, y_{l-1})}\textbf{)} 
\end{flalign}$$

---


### Conditional Random Field (CRF)

There is an effective way to compute $log(Z(\textbf{x}))$ with a complexity of $\mathcal{O}(L)$ using [the Log-Sum-Exp trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/).

$$\begin{flalign}
log(Z(\textbf{x})) &= log(\sum_{n'=1}^{N}exp[{\sum_{l=2}^{L}\textbf{(}U(\textbf{x}, y^{(n')}_{l}) + T(y^{(n')}_{l}, y_{l-1})}\textbf{)}])\\

&= c + log(\sum_{n'=1}^{N}exp[{\sum_{l=2}^{L}\textbf{(}U(\textbf{x}, y^{(n')}_{l}) + T(y^{(n')}_{l}, y_{l-1})}\textbf{)} - c])
\end{flalign}$$

---


### Conditional Random Field (CRF)

If we fix $c = max\{U(\textbf{x}, y^{(1)}_{l}) + T(y^{(1)}_{l}, y_{l-1}), ..., U(\textbf{x}, y^{(N)}_{l}) + T(y^{(N)}_{l}, y_{l-1})\}$ we ensure that the largest positive exponentiated term is $exp(0)=1$.

---


<!--footer: "Course 7: Advanced NLP Tasks" -->
<!--_class: lead -->
## Sentiment Analysis

---


<!--footer: "Sentiment Analysis" -->
### Sentiment Analysis

**Sentiment analysis** is a sentence classification task aiming at **automatically mapping data to their sentiment**.

It can be **binary** classification (e.g., positive or negative) or **multiclass** (e.g., enthusiasm, anger, etc)

---


### Sentiment Analysis

<center><img height="500px" src="https://media.geeksforgeeks.org/wp-content/uploads/20230802120409/Single-Sentence-Classification-Task.png"/></center>

---


### Sentiment Analysis

The loss can be the likes of cross-entropy (CE), binary cross-entropy (BCE) or KL-Divergence (KL).

$$\mathcal{L}_{CE} = - \frac{1}{N} \sum_{n'=1}^{N}y^{(n)}.log(f(\textbf{x}, \theta)^{(n)})$$

$$\mathcal{L}_{BCE} = - y^{(n)}.log(f(\textbf{x}, \theta)^{(n)}) + (1 - y^{(n)}).(1 - f(\textbf{x}, \theta)^{(n)})$$

$$\mathcal{L}_{KL} = - \frac{1}{N} \sum_{n'=1}^{N}y^{(n)}.log(\frac{y^{(n)}}{f(\textbf{x}, \theta)^{(n)}})$$

---


<!--footer: "Course 7: Advanced NLP Tasks" -->
<!--_class: lead -->
## Question Answering (QA)

---


<!--footer: "Question Answering (QA)" -->
### QA

**QA** is the task of **retrieving a span of text from a context** that is best suited to answer a question.

This task is extractive, and can be seen as information retrieval (more on that later).

---


### QA

<center><img height="500px" src="https://scaleway.com/cdn-cgi/image/width=3840/https://www-uploads.scaleway.com/blog-squadbert.webp"/></center>

---

### QA

The loss is the cross entropy over the output of the starting token and the ending one:

$$\mathcal{L}_{CE_{QA}} = \mathcal{L}_{CE_{start}} + \mathcal{L}_{CE_{end}}$$

---


<!--footer: "Course 7: Advanced NLP tasks" -->
<!--_class: lead -->
## Natural Language Inference (NLI)

---


<!--footer: "Natural Language Inference (NLI)" -->
### NLI

**NLI** is the task of **determining whether a "hypothesis" is true (entailment), false (contradiction), or undetermined (neutral)** given a "premise".

---


### NLI

<style scoped>section{font-size:30px;}</style>
Premise|Label|Hypothesis
-------|-----|----------
A man inspects the uniform of a figure in some East Asian country.|contradiction|The man is sleeping.
An older and younger man smiling.|neutral|Two men are smiling and laughing at the cats playing on the floor.
A soccer game with multiple males playing.|entailment|Some men are playing a sport.

---


### NLI

<center><img height="500px" src="https://nlp.gluon.ai/_images/bert-sentence-pair.png"/></center>

---


### NLI

The loss is simply the cross entropy or the divergence over the output of the `CLS` token and the true label.

$$\mathcal{L}_{NLI} = \mathcal{L}_{CE_{CLS}}$$

We are trying to compress the information about both sentence in one `CLS` token via attention and decide about their relationship.

Is it possible to help the model infering more information with les text data?

---


<!--footer: "Course 7: Advanced NLP tasks" -->
<!--_class: lead -->
## Going Further: LM as Knowledge Graphs

---


<!--footer: "Going Further: LM as Knowledge Graphs" -->
### Going Further: LM as Knowledge Graphs

<center><img width="1000px" src="https://figures.semanticscholar.org/ad3dfb2514cb0c899fcb9a14d229ff2a6018892f/2-Figure1-1.png"/></center>

---


### Going Further: LM as Knowledge Graphs

<center><img width="1000px" src="https://figures.semanticscholar.org/ad3dfb2514cb0c899fcb9a14d229ff2a6018892f/7-Table1-1.png"/></center>

Improvements are mostly on dataset with few training examples and complicated examples (negations, non-verbal sentences, ...).

---


### Going Further: LM as Knowledge Graphs

This architecture ***involves a KG ready to use beforehead and pre-training from scratch***.

How can we better **perform NLP task without having to retrain or fine-tune** a model?

---


<!--footer: "Course 7: Advanced NLP tasks" -->
<!--_class: lead -->
## Exploit LLMs capacities: Chain-of-thoughts & In context Learning

---


<!--footer: "Exploit LLMs capacities: Chain-of-thoughts & In context Learning" -->
### Exploit LLMs capacities

**ICL** enables LLMs to **learn new tasks** using natural language prompts **without explicit retraining or fine-tuning**.

The **efficacy** of ICL is **closely tied to** the model's **size**, training **data quality**, and **domain specificity**.

---


### Exploit LLMs capacities

<center><img height="500px" src="https://thegradient.pub/content/images/size/w800/2023/04/icl-copy2.png"/></center>

---


### Exploit LLMs capacities

<center><img height="500px" src="https://lh6.googleusercontent.com/In6MiddAKdLNEjwHeOzkIJlK3FmZank8f2ibBERPReIwTAKkDm4HglsizdjE8O23gmjyPaEFJSMsdRZLiVx5vNE6RLY2pyukmSEh9acYSwBCUNljXpcalKK4d0KUvcRNlEsNG7x4Exn7jDOEHDwbyE0"/></center>

---


### Exploit LLMs capacities

<center><img height="500px" src="https://lh6.googleusercontent.com/L_cA-kq0nkDAPO76ju9z8m_3KmZ8nyOIvXrOPoQ9ldAXCR0ACtFOanfCYUllb2g9OBa-2nG5BnsgjKuEPXSlbmgbRNqbS9p3vldqark5wAaTWnGsJofzNzK3GKUsww6byRCgA_AmHcItRgPLoFSk8N0"/></center>

---


### Exploit LLMs capacities

<center><img height="500px" src="https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fzero-cot.79793bee.png&w=1080&q=75"/></center>

---


<!--footer: "Course 7: Advanced NLP tasks" -->
<!--_class: lead -->
## Questions?
