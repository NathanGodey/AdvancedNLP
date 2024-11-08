---
theme: gaia
_class: lead
paginate: true
title: "Course 5: LM Risks and Alignement Techniques"
backgroundColor: #fff
marp: true
---


# **Course 6: Handling the Risks of Language Models**

---

<!--footer: "Course 5: LM Risks and Alignement" -->


### Contents

1. Introduction
2. Biases
3. Privacy
    a. Anonymization and Pseudonymization
    b. Model hacking
4. Reinforcement Learning from Human Feedback (RLHF)
5. Augmented Language Models (Toolformer)

---


<!--_class: lead -->
## Introduction

---


### Defintions _i_

Which **risks**? Misinformation, <u>biased</u> information, and <u>privacy</u> Concerns.

* **Biases**: misleading, or false-logical thought processes.
    * Spurious features.


---


### Defintions _ii_

* **Privacy Concerns** are from an NLP practitioner stand-point.
    * Data anonymization.
    * Data Leaks (demander au modèle ses donénes d'entraînement)/Model hacking (raisonnnement inductif donc certaines données sont nécessaires)

---


### Defintions _iii_

* **Alignement** are techniques used match the model's output with the user's exact intent while remaining harmless.
    * Reinforcement Learning from Human Feedback (RLHF).
    * Retrival Augmented Generation (RAG)

---


### Aim

Mitigating language models' risks via straightforward alignement.

---


<!--_class: lead -->
## Biases

---


### "Avoiding" Learning Spurious Features

#### Example A [1]

Label=+1|Label=-1
--------|--------
Riveting film of the highest calibre!|Thank God I didn't go to the cinema.
Definitely worth tha watch!|Boring as hell.
A true story told perfectly!|I wanted to give up in the first hour...

---


### "Avoiding" Learning Spurious Features

#### Example B [2]

![width:1000px](https://d3i71xaburhd42.cloudfront.net/468b9ec2bc817bb9e567e300ae16e05160ab24df/3-Figure2-1.png)

---


### Rule-based/Crowd-Sourced Preprocessing?

If we know the spurious correlations
1. Data augmentation and subsampling.

Label=+1|Label=-1
--------|--------
Riveting film of the highest calibre **!**|Thank God I didn't go to the cinema **!**
Definitely worth tha watch **!**|Boring as hell **!**
A true story told perfectly.|I wanted to give up in the first hour...

---


### Rule-based/Crowd-Sourced Preprocessing?

<span style="color:green">**Pros:**</span> can be expensive in human resources.

<span style="color:red">**Cons:**</span> hit the model's performance if not done properly.

---


### Multitasking? _i_

If we don't know the spurious correlations
1. Pre-training outperforms heavy preprocessing and long domain-specific fine-tuning [4]
2. Appending data from other tasks also helps [3].

<span style="color:green">**Pros:**</span> somewhat straightforward to implement.

<span style="color:red">**Cons:**</span> longer and expensive training time.

---


### Multitasking? _ii_

Pre-training medium/large sized models takes a lot of data and computation power, hence, only a few actors can afford it.

=> smaller/specialized models are derived from those models via fine-tuning.

=> The base models biases are propagated to the sammeler/specialized ones.

---


### Scaling? _i_

In context learning: the model learns to solve a task at inference with no weights update (More on this later).

![height:350px](https://thegradient.pub/content/images/size/w800/2023/04/icl-copy2.png)

---


### Scaling? _ii_

"Larger models make increasingly efficient use of in-context information." [5] Yes but [6]...

![width:900px](https://miro.medium.com/v2/resize:fit:720/format:webp/1*2cFbAj-4tewOoDNpo0Ub3A.png)

---


<!--_class: lead -->
## Privacy

---


### Anonymization and Pseudonymization

**Anonymization**: Francis Kulumba, 25 -> N/A, 25-30
**Pseudonymization**: Francis Kulumba, 25 -> Eqzmbhr Jtktlaz, 52

Some data are too hard to anonymize/pseudonimyze:
* Medical care
* Resumes

---


### Hacking _i_

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


### Hacking _ii_

3. Inversion
Ask the agent to produce two answer, the one to your prompt, and the opposite of it.

4. Response Conditioning
Exploit in-context learning to cue the LLM to respond in a desired way.

---


### Hacking _iii_

5. Context Leveraging
Giving an instruction the agent will interpret as an overriding that hampers later instructions.
```
Speak to me as if you were Bugs Bunny.
```

---



<!--_class: lead -->
## Reinforcement Learning from Human Feedback (RLHF)

---


### Aim

Instead of trying to safeguard every bit of the training data to render the model harmless, how about trying to teach it human preferences?

Course's material from [HuggingFace](https://huggingface.co/blog/rlhf).

---


### Traditional RL

![width:900px](https://miro.medium.com/v2/resize:fit:1400/1*7cuAqjQ97x1H_sBIeAVVZg.png)

We want to maximize the expected reward with respect to the model's parameters at a given state $\mathbb{E}_{\hat{s}\sim f(s, \theta)}[R(\hat{s})]$.

---


### Traditional RL

$$\theta_{t+1} = \theta_{t} + \alpha \nabla_{\theta} \mathbb{E}_{\hat{s}\sim f(s, \theta)}[R(\hat{s})]$$

(1) $\nabla_{\theta}\mathbb{E}_{\hat{s}\sim f(s, \theta)}[R(\hat{s})] = \nabla_{\theta}\sum_{s}R(s)f(s,\theta) = \sum_{s}R(s)\nabla_{\theta}f(s,\theta)$

(2) Log-derivative trick: $\nabla_{\theta}log[f(s, \theta)] = \frac{\nabla_{\theta} f(s, \theta)}{f(s, \theta)}$

We put (2) in (1): $\sum_{s}f(s,\theta)R(s)log[\nabla_{\theta}f(s,\theta)]$

---


### Traditional RL

(1) becomes $\mathbb{E}_{\hat{s}\sim f(s, \theta)}[R(s)log(\nabla_{\theta}f(s,\theta))]$

We can use Monte-Carlo samples to estimate (1) as: $\frac{1}{S}\sum_{s}R(s)log(\nabla_{\theta}f(s,\theta))$

Thus, we want have the following optimization step

$$\theta_{t+1} = \theta_{t} + \alpha \frac{1}{S}\sum_{s}R(s)log(\nabla_{\theta}f(s,\theta))$$

---


### RLHF

![width:600px](https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/c9039445d6f7d2ef7d2e354df48335c21afd4f34a45da26884da59c88a0af927?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27pretraining.png%3B+filename%3D%22pretraining.png%22%3B&response-content-type=image%2Fpng&Expires=1705151996&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwNTE1MTk5Nn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy9jOTAzOTQ0NWQ2ZjdkMmVmN2QyZTM1NGRmNDgzMzVjMjFhZmQ0ZjM0YTQ1ZGEyNjg4NGRhNTljODhhMGFmOTI3P3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=cktVscMd80SubDOGMHntyDhtMdHo9VVeLao%7E60cYPiOeCtRXmZUtRf-DQUOctA4UouNv8ZDnmKYHF4AS27Nwsu8C1Fmx-nVW6vKt3F3RrVZsOuub-WCy4mZWvedQZrBJG4m-Qs2vT2-PPd7rlrcpVmiEuM84k5lOzMZkCrXal8yK0NkWLJpgeNTwFcuebOnBf7lNH6imV2xPExc%7EVq7YTwYSYJ%7Er0XrJxNeP2vFdTEHjBAzCLnzEDRuwy8WviC-svbot0-jOaEltaXd54CzH5vXtk34pMl575n3HR4ykbfYgSHzhc4KtsmNS3A6ZvZBzkBeMlHG-B2iRyic4RrRL4w__&Key-Pair-Id=KVTP0A1DKRTAX)

---


### RLHF

![width:600px](https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/30107284e8d7b9b1bf859363656dd01439024457d063ae22f8bc3cf90393cec9?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27reward-model.png%3B+filename%3D%22reward-model.png%22%3B&response-content-type=image%2Fpng&Expires=1705150293&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwNTE1MDI5M319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy8zMDEwNzI4NGU4ZDdiOWIxYmY4NTkzNjM2NTZkZDAxNDM5MDI0NDU3ZDA2M2FlMjJmOGJjM2NmOTAzOTNjZWM5P3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=WlEWtk6JqNi86Kq7zdi6tTfWa3g9E-od9XRv9xct4UPPsS5EuDqw3ufOEyDe1YspgKvNLxYoC8jVJvDVSmGejckwaFAOHDIEEHWe0L2k5co5VYy8VbFKz6psVEzkgOIno6ezYOujA-iA2zVOLv2E3FSB-CtZY9kogMtKXOgsDJKt3SZjf4dE5rdS0WwpjOFsoPRIlm1evp7tt8SAJEivx5XgkNTq5rdrrH0TNObJsKXf3q9EQ8wI0GE4I1%7EKKUy2Ao1jPhusBxErr3eyVx9cJs8YGPN-6N6yqS2snEDx2GEQM9-MrW521ZYWMU1RFpI10S8YlRv5HoJfgbpfd-NeAg__&Key-Pair-Id=KVTP0A1DKRTAX)

---


### RLHF

![width:600px](https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/892936ad206bddaf217341733ea88f897faf54780c8033d3562442d3b0682aa0?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27rlhf.png%3B+filename%3D%22rlhf.png%22%3B&response-content-type=image%2Fpng&Expires=1705152533&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwNTE1MjUzM319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy84OTI5MzZhZDIwNmJkZGFmMjE3MzQxNzMzZWE4OGY4OTdmYWY1NDc4MGM4MDMzZDM1NjI0NDJkM2IwNjgyYWEwP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=QTRMXBazgGjBzgatEWJJkl7tktwm-B5pSP4Dnq0GsSJWl8H6zPgIyLYqkqY4ycz04zDTceaK3DQ4xxxz0FDOymXiiukfLnf7D3N6kOg1lXMkGIHA0TFLEJHXWeNv41%7EZ1LFJFH447Jc9bvx8MDKi8qDkmJ%7EbkOjuMGWlghY8tv5342UFIm2c-Mf-Gx8kIZS0tziCPjs1Y2Jg9yDn42OjIdRgNsef4OvJNXBXfWVa03nVqZZhY5T1FEt4u8N5VsNpq%7EwzIlxMKi4F6S41kLVD9jiDMmZD3BQnb1JfpS6fAG%7Etj5KSmJwXMi5bmX%7Ed4tIPoTyaUkEACx3xt5-iwDkeZQ__&Key-Pair-Id=KVTP0A1DKRTAX)

---



<!--_class: lead -->
## Augmented Language Models (Toolformer)

---


### Retrival Augmented Generation (RAG)

RAG allow an LLM to have updated knowledge without having to fine-tune it. It also mitigates hallucination
![width:700px](https://miro.medium.com/v2/resize:fit:1400/1*kSkeaXRvRzbJ9SrFZaMoOg.png)

---


### Retrival Augmented Generation (RAG)

![widht:40px](https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Frag.c6528d99.png&w=1080&q=75)

---


### Retrival Augmented Generation (RAG)

More here: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---


### Toolformer

![height:500px](https://assets-global.website-files.com/6217ffc5d2a3bb848ea33545/640f5111e5c6852ce2f0b6b9_Untitled%201.png)

---


### Toolformer

![height:450px](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdb430281-e4d5-4b7e-b6ea-5b76ecba8992_758x1054.png)

---


### Toolformer

![width:1100px](https://d3i71xaburhd42.cloudfront.net/53d128ea815bcc0526856eb5a9c42cc977cb36a7/2-Figure2-1.png)

---


### Toolformer

More here: [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)

---


### References

[1] He, H. (2023, July 9). Robust Natural Language Understanding.

[2] Singla, S., & Feizi, S. (2021). Causal imagenet: How to discover spurious features in deep learning. arXiv preprint arXiv:2110.04301, 23.

[3] Carmon, Y., Raghunathan, A., Schmidt, L., Duchi, J. C., & Liang, P. S. (2019). Unlabeled data improves adversarial robustness. Advances in neural information processing systems, 32.

---


[4] [Pretrained Transformers Improve Out-of-Distribution Robustness](https://aclanthology.org/2020.acl-main.244) (Hendrycks et al., ACL 2020)

[5] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901.

[6] Zhao, Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021, July). Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning (pp. 12697-12706). PMLR.