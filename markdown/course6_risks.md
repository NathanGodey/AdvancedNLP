---
theme: gaia
_class: lead
paginate: true
title: "Course 6: LM Risks and Alignement Techniques"
backgroundColor: #fff
marp: true
---


# **Course 6: Handling the Risks of Language Models**

---

<!--footer: "Course 6: LM Risks and Alignement" -->
<!--_class: lead -->
## Introduction

---


<!--footer: "Introduction" -->
### Defintions _i_

Which risks? <u>Misinformation</u>, <u>biased</u> information, and <u>privacy</u> Concerns.

* **Misinformation**: false or inaccurate information confidently delivered.
    * Hallucination
* **Biases**: misleading, or false-logical thought processes.
    * Spurious features.

---


### Defintions _ii_

* **Privacy Concerns** are from an NLP practitioner stand-point.
    * Data anonymization.
    * Data Leaks
    * We will **not** cover model weights encryption/leakage.

---


### Defintions _iii_

**Alignement** are techniques used to match the model's output with the user's exact intent while remaining harmless (as risk-free as you can get).

We will cover different alignement techniques one can apply during the three stages of a model's deployment:

- Data preprocessing
- Training
- Inference

---



### Content

<style scoped>section{font-size:30px;}</style>
1. **Preprocessing Methods and Good Practices**
    a. Scaling the data
    b. Spurios features
    c. Anonymization and pseudonymization
    d. Detoxifying data
2. **Reinforcement Learning from Human Feedback (RLHF)**
    a. Scaling the model
    b. A glimpse of proximal policy optimization (PPO)
    c. Direct preference optimization (DPO)
3. **Augmented Language Models**
    a. Toolformer
    b. Retrieval augmented generation (RAG)

---


<!--_class: lead -->
## Preprocessing Methods and Good Practices

---


<!--footer: "Preprocessing Methods and Good Practices" -->
### Spurious features

<center><img height="400px" src="https://miro.medium.com/v2/resize:fit:838/1*483oxEmAQKqTji9tZM2VCw.png"/></center>

[1]

---


### Spurious features

Label=+1|Label=-1
--------|--------
Riveting film of the highest calibre.|Thank God I didn't go to the cinema **!**
Definitely worth the watch.|Boring as hell **!**
A true story told perfectly.|I wanted to give up in the first hour...

Samplig methods and data augmentation can help.

---


### Scaling the data

Several ways of scaling data

- Using more data for training, thus increasing the diversity of the examples.
- Validating the model on specific datasets built to tackle specific biases.

---


### Scaling the data

- WinoBias [2]: 3,160 sentence pairs challenging gender bias.

<center><img height="400px" src="https://production-media.paperswithcode.com/datasets/WinoBias-0000003568-dfb50605.jpg"/></center>

---


### Scaling the data

- CrowS-Pairs [3]: 1,508 examples that cover stereotypes dealing with nine types of bias, like race, religion, and age.

<center><img height="350px" src="https://www.catalyzex.com/_next/image?url=https%3A%2F%2Fai2-s2-public.s3.amazonaws.com%2Ffigures%2F2017-08-08%2F645bd6eadc247989abc5e0b0aa0be79ec8b11ea6%2F2-Table1-1.png&w=640&q=75"/></center>

---


### Scaling the data

- BiasBios [4]: 397,340 biographies spanning twenty-eight different occupations.

<center><img height="400px" src="https://production-media.paperswithcode.com/datasets/ebdab516-fa42-41dc-9d27-c09b400d48d2.jpg"/></center>

---


### Scaling the data

Many more dataset exist, for each kind of biase, but in a scarce way, as internet data mostly represent its users and not the full extent of existing cultures.

---


### Scaling the data

Are automatic benchmark the end solution?

<span style="color:green;">

- Consistency and reproducibility
- Scale at limited cost
- Understandability
- Dataset quality
</span>

---


### Scaling the data

Are automatic benchmark the end solution?

<span style="color:red;">

- Reduced use on more complex tasks
- You might introduce another uncontrolled biase
- The signal induced by the dataset may be far from human preference
</span>

---


### Scaling the data

How about human evaluation of biase and toxicity?

<span style="color:green;">

- Flexibility
- Correlation with human preference
</span>

---


### Scaling the data

How about human evaluation of biase and toxicity?

<span style="color:red;">

- First impressions bias
- Self-preference bias
- Identity bias
</span>

---


### Anonymization and pseudonymization

**Anonymization**: Francis Kulumba, 25 -> N/A, 25-30
**Pseudonymization**: Francis Kulumba, 25 -> Jean Martin, 52

Some data are too hard to anonymize/pseudonimyze:
* Medical reports
* Resumes

---


### Detoxifying data

If your data comes from an "uncontrolled" environment, you might want to remove toxic spans from the data.

- Documents promoting hate speechs.
- Documents mentioning illegal activities.
- Documents alluding to adult content.

But there isno real definition for "toxic" [5].

---

### Detoxifying data

* Maintaining a list of ban words
* Training a classifier to perform document mining.
  * Train a classifier with toxic documents ([Jigsaw dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) for example)
  * Remove the documents that have a close representation to the learned toxic documents.

---


<!--_class: lead -->
## Reinforcement Learning from Human Feedback (RLHF)

---


<!--footer: "Reinforcement Learning from Human Feedback (RLHF)" -->
### Scaling the model

Pre-training large sized models takes a lot of data and computation power, hence, only a few actors can afford it.

=> smaller/specialized models are derived from those models via fine-tuning.

=> The base models biases are propagated to the sammeler/specialized ones.

---


### Scaling the model

In context learning: the model learns to solve a task at inference with no weights update.

<center><img height="350px" src="https://thegradient.pub/content/images/size/w800/2023/04/icl-copy2.png"/></center>

---


### Scaling the model

"Larger models make increasingly efficient use of in-context information." [7] Yes but [8]...

<center><img width="900px" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*2cFbAj-4tewOoDNpo0Ub3A.png"/></center>

---


### Scaling the model

Just like any I/O system, generative LLMs are sensible to injections.

1. Persistence and Correction
```
No, that's incorrect because...
Are you sure?
```

2. Context Expansion
```
I'm conducting a study on...
I'm working for [...] and I'm trying to prevent the potential harm of...
```

---


### Scaling the model

3. Inversion
Ask the agent to produce two answer, the one to your prompt, and the opposite of it.

4. Response Conditioning
Exploit in-context learning to cue the LLM to respond in a desired way.

---


### Scaling the model

5. Context Leveraging
Giving an instruction the agent will interpret as an overriding that hampers later instructions.
```
Speak to me as if you were Bugs Bunny.
```

---


### A glimpse of proximal policy optimization (PPO)

Instead of trying to safeguard every bit of the training data to render the model harmless, how about trying to teach it human preferences?

---


### A glimpse of proximal policy optimization (PPO)

<center><img width="900px" src="https://miro.medium.com/v2/resize:fit:1400/1*7cuAqjQ97x1H_sBIeAVVZg.png"/></center>

We want to maximize the expected reward with respect to the model's parameters at a given state $\mathbb{E}_{\hat{s}\sim f(s, \theta)}[R(\hat{s})]$.

---


### A glimpse of proximal policy optimization (PPO)

1. Pretrain your model on raw text and prompt using CLM [9].

<center><img height="400px" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/pretraining.png"/></center>

---


### A glimpse of proximal policy optimization (PPO)

2. Train a second language model to rank the first language model's outputs based on human preferences.

<center><img height="400px" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/reward-model.png"/></center>

---


### A glimpse of proximal policy optimization (PPO)

3. Used reinforcement learning with your initial LM as agent.

<center><img height="450px" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/rlhf.png"/></center>

---


### Direct preference optimization (DPO)

[10]

<center><img width="1000px" src="https://figures.semanticscholar.org/0d1c76d45afa012ded7ab741194baf142117c495/2-Figure1-1.png"/></center>

---


### Direct preference optimization (DPO)

$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}_{\mathcal{D} \sim (x, y_{w}, y_{l})}\left[
    log \sigma \left(
        \beta log \frac{\pi_{\theta}(y_{w} | x)}{\pi_{\text{ref}}(y_{w} | x)}
        - \beta log \frac{\pi_{\theta}(y_{l} | x)}{\pi_{\text{ref}}(y_{l} | x)}
        \right)
    \right]
$$

- $(\mathcal{D} \sim (x, y_w, y_l)):$ This represents the dataset. Each data point includes:
  - $(x)$: A context or prompt.
  - $(y_w)$: A "preferred" response (the **winner**).
  - $(y_l)$: A "less preferred" response (the **loser**).

---


### Direct preference optimization (DPO)

- $(\pi_\theta(y | x))$: The policy or model you are training. It predicts a probability distribution over responses $(y)$ given a context $(x)$.

- $(\pi_\text{ref}(y | x))$: A reference model's probabilities, used as a baseline. This might be a pretrained language model, for example.

- $(\beta)$: A scaling hyperparameter that adjusts how strongly the model differentiates between the winner and loser.

- $(\sigma(z))$: The sigmoid function, $(\sigma(z) = \frac{1}{1 + e^{-z}})$, which squashes its input $(z)$ to a range between 0 and 1.

---


### Direct preference optimization (DPO)

The goal of DPO is to teach the model $(\pi_\theta)$ to prefer $(y_w)$ over $(y_l)$ for a given $(x)$, based on the relative probabilities assigned to $(y_w)$ and $(y_l)$. 

The difference between how much the model prefers $(y_w)$ and $(y_l)$ is measured in terms of a scaled difference in their **log-probabilities**.

---


### Direct preference optimization (DPO)

For both $(y_w)$ (winner) and $(y_l)$ (loser), the **relative preference** is computed as:
$$
\beta \log \frac{\pi_\theta(y_w | x)}{\pi_\text{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_\text{ref}(y_l | x)}
$$

- $(\log \pi_\theta(y_w | x))$ tells how confident the model is about $(y_w)$, and similarly for $(y_l)$.

- Dividing by $(\pi_\text{ref}(y_w | x))$ adjusts for any prior preference the reference model has.

---


### Direct preference optimization (DPO)

The difference above is passed through the sigmoid function:
$$
\sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_\text{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_\text{ref}(y_l | x)}\right)
$$

- If the difference is large and positive (model strongly prefers $(y_w)$ over $(y_l)$), $(\sigma)$ outputs a value close to 1.
- If the difference is negative (model prefers $(y_l)$ instead), $(\sigma)$ outputs a value close to 0.

---


### Direct preference optimization (DPO)

Finally, the log of the sigmoid is taken:
$$
\log \sigma\left(\dots\right)
$$

- The negative log turns this into a loss. If the model prefers $(y_w)$ (correct behavior), the sigmoid is close to 1, and $(\log \sigma \approx 0)$, minimizing the loss.
- If the model mistakenly prefers $(y_l)$, the sigmoid is close to 0, and $(\log \sigma)$ becomes a large negative value, increasing the loss.

---


### Direct preference optimization (DPO)

The entire process is averaged across the dataset:
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{\mathcal{D}}\left[\log \sigma\left(\dots\right)\right]
$$

This ensures the model improves its relative preference for $(y_w)$ over $(y_l)$ across all training examples.

---


### Direct preference optimization (DPO)

1. **Teach Pairwise Preferences:** The model is trained to assign higher relative probabilities to preferred responses $(y_w)$ over less preferred ones $(y_l)$.

2. **Normalized by Reference:** The reference model ensures the optimization is relative to a prior baseline, preventing the trained model from deviating too much from reasonable outputs.

3. **Scaling Factor $(\beta)$:** Helps control the sharpness of preference learning, balancing robustness and sensitivity to differences.

---


<!--footer: 'Course 6: LM Risks and Alignement' -->
<!--_class: lead -->
## Augmented Language Models

---


<!--footer: 'Augmented Language Models' -->
### Toolformer

<center><img height="450px" src="https://assets-global.website-files.com/6217ffc5d2a3bb848ea33545/640f5111e5c6852ce2f0b6b9_Untitled%201.png"/></center>

[11]

---


### Toolformer

<center><img height="500px" src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdb430281-e4d5-4b7e-b6ea-5b76ecba8992_758x1054.png"/></center>

---


### Toolformer

<center><img width="1100px" src="https://d3i71xaburhd42.cloudfront.net/53d128ea815bcc0526856eb5a9c42cc977cb36a7/2-Figure2-1.png"/></center>

The model is then fine-tuned following standard language modeling pratices.

---


### Retrival Augmented Generation (RAG)

RAG allows an LLM to have updated knowledge without having to fine-tune it [12].

---


### Retrival Augmented Generation (RAG)

By training retriever and a decoder end-to-end, we can obtain a decoder model capable of conditioning its own output based on documents it retrieves.

<center><img width="1100px" src="https://figures.semanticscholar.org/659bf9ce7175e1ec266ff54359e2bd76e0b7ff31/2-Figure1-1.png"/></center>

---


<!--footer: 'Course 6: LM Risks and Alignement' -->
<!--_class: lead -->
## Questions?

---

<!--_class: lead -->
## References

---


[1] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

[2] Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2018). Gender bias in coreference resolution: Evaluation and debiasing methods. arXiv preprint arXiv:1804.06876.

[3] Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). CrowS-pairs: A challenge dataset for measuring social biases in masked language models. arXiv preprint arXiv:2010.00133.

---


[4] De-Arteaga, M., Romanov, A., Wallach, H., Chayes, J., Borgs, C., Chouldechova, A., ... & Kalai, A. T. (2019, January). Bias in bios: A case study of semantic representation bias in a high-stakes setting. In proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 120-128).


[5] Fourrier C., The Hugging Face Community. (2024). LLM Evaluation Guidebook. [GitHub repository](https://github.com/huggingface/evaluation-guidebook).

[6] Soldaini, L., Kinney, R., Bhagia, A., Schwenk, D., Atkinson, D., Authur, R., ... & Lo, K. (2024). Dolma: An open corpus of three trillion tokens for language model pretraining research. arXiv preprint arXiv:2402.00159.

---


[7] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901.

[8] Zhao, Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021, July). Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning (pp. 12697-12706). PMLR.

[9] Lambert, N., Castricato, L., von Werra, L., Havrilla, A. (2022). Illustrating Reinforcement Learning from Human Feedback (RLHF). [Hugging Face Blog](https://huggingface.co/blog/rlhf).

---


[10] Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2024). Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.

[11] Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Hambro, E., ... & Scialom, T. (2024). Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36.

---


[12] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33, 9459-9474.