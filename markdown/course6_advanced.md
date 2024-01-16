---
theme: gaia
_class: lead
paginate: true
title: "Course 6: Advanced NLP tasks"
backgroundColor: #fff
marp: true
---


# **Advanced NLP tasks**

---

<!--footer: "Course 6: Advanced NLP tasks" -->


### Contents

1. Named Entity Recognition (NER)
    a. Part-of-Speech Tagging (POS)
    b. Conditional Random Field (CRF)
    c. Weakly Supervised NER
2. Sentiment Analysis
3. Natural Language Inference (NLI)
4. QuestionAnswering (QA)
    a. Going further: LM as knowledge graphs
5. Exploit LLMs capacities: Chain-of-thoughts & In context learning

---



<!--_class: lead -->
## Named Entity Recognition (NER)

---


### NER

Named entity recognition (NER), aims at identifying real-world entity mentions from texts, and classifying them into predefined types.

![height:300px](../imgs/course6/ner_example.png)

---


### NER

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

Knowing that language models are good at generating vector spaces to better represent words:
for each token in a sentence we want to compute a probability $p$ to belong to a class $n$.

$$p: f(\textbf{x}, \theta)_{l} \mapsto ?$$
with $p \in [0, 1]$
Where $f(\textbf{x}, \theta)_{l}$ are the language model's parameters for the $l_{1 \leq L}$-th token.

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

![height:450px](https://raw.githubusercontent.com/PythonWorkshop/intro-to-nlp-with-pytorch/master/images/viterbi.png)

---


### Conditional Random Field (CRF)

![height:450px](https://raw.githubusercontent.com/PythonWorkshop/intro-to-nlp-with-pytorch/master/images/crf_transition_matrix.png)

---


### Conditional Random Field (CRF)

![height:450px](https://raw.githubusercontent.com/PythonWorkshop/intro-to-nlp-with-pytorch/master/images/linear_crf_example.png)

---


### Conditional Random Field (CRF)

What do we learn?

---




<!--_class: lead -->
## Questions?

---


### References

[1] He, H. (2023, July 9). Robust Natural Language Understanding.

[2] Singla, S., & Feizi, S. (2021). Causal imagenet: How to discover spurious features in deep learning. arXiv preprint arXiv:2110.04301, 23.

[3] Carmon, Y., Raghunathan, A., Schmidt, L., Duchi, J. C., & Liang, P. S. (2019). Unlabeled data improves adversarial robustness. Advances in neural information processing systems, 32.

---


[4] [Pretrained Transformers Improve Out-of-Distribution Robustness](https://aclanthology.org/2020.acl-main.244) (Hendrycks et al., ACL 2020)

[5] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901.

[6] Zhao, Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021, July). Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning (pp. 12697-12706). PMLR.