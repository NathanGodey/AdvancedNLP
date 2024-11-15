---
theme: gaia
_class: lead
paginate: true
title: "Course 3: Language Modeling"
backgroundColor: #fff
marp: true
---

# **Course 3: Language Modeling**


---
<!--footer: 'Course 3: Language Modeling' -->

<center>
<img width="900px" src="../imgs/course3/chatgpt.png"/>

**How does it really work?**
</center>

---
<!--_class: lead -->
# What is Language Modeling?

---

### Definition

- A sequence of tokens $(w_1, w_2, ..., w_n)$
- For a position $i$, a language model (**LM**) predicts
$$
P(w_i\ | (w_j)_{j\neq i}) \in \Delta^V
$$

- In words: a LM predicts the probability of a token given its context

---

### Example

*I went to the **???** yesterday*

P(*park* |  *I went to the **???** yesterday*) = 0.1

P(*zoo* |  *I went to the **???** yesterday*) = 0.07

...

P(*under* |  *I went to the **???** yesterday*) = 0

---

### Why is it hard?

* **Large vocabularies**: 170,000 English words
* **Lots of possible contexts**:
    * For $V$ possible tokens, there are $V^L$ contexts of size $L$ (in theory)
* **Inherent uncertainty**: not obvious even for humans

---

### Basic approach - Unigram

- Learn the *non-contextual* probability (=frequency) of each token:
$$
P(w_i\ | (w_j)_{j\neq i}) = f
$$

**Example**
*chart against operations at influence the surface plays crown a inaro the three @ but the court lewis on hand american of seamen mu role due roger executives*

---

### Include context - Bigram

- Predict based on the last token only:
$$
P(w_i\ | (w_j)_{j\neq i}) = P_{\theta}(w_i | w_{i-1})
$$

- (MLE): Measure next token frequency

**Example**
*the antiquamen lost to dios nominated former is carved stone oak were problematic, 1910. his willingness to receive this may have been seen anything*

---

### Include more context - n-gram

- Predict based on the $n$ last tokens only:
$$
P(w_i\ | (w_j)_{j\neq i}) = P_{\theta}(w_i | w_{i-n}...w_{i-1})
$$

- (MLE): Measure occurences of tokens after $w_{i-n}...w_{i-1}$

**Example (n=4)**
*eva gauthier performed large amounts of contemporary french music across the united states marshals service traveled to frankfurt, germany and took custody of the matthews*

---

### Statistical n-grams: pro/cons

* Strenghts:
    * Easy to train
    * Easy to interpret
    * Fast inference
* Limitations:
    * Very limited context
    * **Unable to extrapolate** : can only model what it has seen

---

### The embedding paradigm
<center><img width="600px" src="../imgs/course1/embeddings.png"/></center>

---

### LM with RNNs

<center><img width="700px" src="../imgs/course3/RNN-LM.svg"/></center>

---

### LM with RNNs - Training

- $\theta$: parameters of the RNN
- $(w_1,...,w_n)$: training sequence
- Cross-entropy loss $\mathcal{L}_{ce}$:
$$
\mathcal{L}_{ce}(w, \theta) = - \sum_{i=2}^{n} 1_{w_i} \cdot \log P_{\theta}(w_i |w_{i-1}, h_{i-1})
$$
- Train via back-propagation + SGD

---

### Reminder - Back-propagation

<center><img width="1100px" src="../imgs/course3/backprop.gif"/></center>

---

### Reminder - Stochastic Gradient Descent

* **Goal** : Minimize a loss function $\mathcal{L}(X, \theta)$ for given data $X$ with respect to model parameters $\theta$

* **Method** : 
    * Split $X$ in smaller parts $x^i$ (called mini-batches)
    * Compute $\mathcal{L}(x^i, \theta)$ (forward) and $\nabla_{\theta} \mathcal{L}(x^i, \theta)$ (back-prop)
    * Update: $\theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}(x^i, \theta)$ &nbsp; &nbsp;  &nbsp; &nbsp; ($\eta \ll 1$, learning rate)

---

### LM with RNNs: Generation

<center><img width="600px" src="../imgs/course3/gen_RNN-LM.svg"/></center>

---

### RNNs: pro/cons
* Strenghts
    * Still relatively fast to train
    * ... and for inference ($O(L)$)
    * **Can extrapolate** (works with continuous features)
* Limitations
    * **Context dilution** when information is far away

---

### Extending RNNs: BiLSTMs
- LSTM: improves context capacity
- Read the sequence in both directions
<center><img width="1000px" src="../imgs/course3/bilstm.png"/></center>

---
<!--_class: lead -->
# Transformers
---
### Information flow - RNN
How many steps between source of info and current position?
* *What is the previous word?* => $O(L)$
* *What is the subject of verb X?* => $O(L)$
* *What are the other occurences of current word?* => $O(L^2)$
* ...
---
### Information flow - Transformers
How many steps between source of info and current position?
* *What is the previous word?* => $O(1)$
* *What is the subject of verb X?* => $O(1)$
* *What are the other occurences of current word?* => $O(1)$
* ... => $O(1)$

---
### Outside Transformers
* A Transformer network $T_{\theta}$
* Input: Sequence of vectors $(e_1,...,e_n) \in \mathbb{R}^D$
* Output: Sequence of vectors $(h_1,...,h_n) \in \mathbb{R}^D$
* Each $h_i$ may depend on the <ins>whole</ins> input sequence $(e_1,...,e_n)$

---
### Inside Transformers
<center><img width="300px" src="../imgs/course3/transformers.png"/></center>

---
### Inside Transformers : Embeddings
Before going in the network:
* Given an input token sequence $(w_1,...,w_n)$
* We retrieve token embeddings $(e_w(w_1),..., e_w(w_n)) \in \mathbb{R}^D$
* We retrieve position embeddings $(e_p(1),..., e_p(n))\in \mathbb{R}^D$
* We compute input embeddings: $e_i = e_w(w_i) + e_p(i)$

---
### Inside Transformers : Self-attention
<center><img width="700px" src="../imgs/course3/transformers_self_attn_qkv.svg"/></center>

---
### Inside Transformers : Q and K
=> Model interactions between tokens:
<center><img width="500px" src="../imgs/course3/transformers_self_attn_qk_prod.svg"/></center>

---
### Inside Transformers : Q and K
- Each row of $QK^T$ is then normalized using softmax
- Interpretable patterns:
<center><img width="400px" src="../imgs/course3/attn_example.png"/></center>

---
### Inside Transformers : Q and K
- Formally:
$$
A_{i,j} = \frac{1}{\sqrt{d_h}} \cdot \frac{e^{(QK^T)_{i,j}}}{\sum_{k}e^{(QK^T)_{i,k}}}
$$

where $d_h$ is the hidden dimension of the model

---
### Inside Transformers : A and V
<center><img width="700px" src="../imgs/course3/self_attn_av.svg"/></center>

---
### Inside Transformers : Self-attention summary

<div style="display: flex;">

  <!-- Left half (image) -->
  

  <!-- Right half (bullet points) -->
  <div style="flex: 1; padding: 0 20px;">
    <ul>
      <li>Inputs are mapped to Queries, Keys and Values</li>
      <li>Queries and Keys are used to measure interaction (A)</li>
      <li>Interaction weights are used to "select" relevant Values combinations</li>
      <li><b>Complexity: O(L^2)</b></li>
    </ul>
  </div>
  <div style="flex: 0.3;">
    <img width="350px" src="../imgs/course3/transformers_fullpic.svg"/>

  </div>

</div>

---
### Inside Transformers : Multi-head attention

<center><img width="800px" src="../imgs/course3/multi_head.svg"/></center>

---
### Inside Transformers : LayerNorm
- Avoids gradient explosion

<center><img width="700px" src="../imgs/course3/layer_norm.png"/></center>

---
### Inside Transformers : Output layer
<br>
<center><img width="400px" src="../imgs/course3/final_proj.svg"/></center>

---
### Modern flavors : Relative Positional Embeddings
- Encode position at attention-level:
$$
(\Omega Q K^T)_{i, j} = \langle \omega_i(Q_i) , \omega_j(K_j) \rangle + \beta_{i, j}
$$
- Rotary Positional Embeddings (RoPE, Su et al. 2023)
  - $\omega_i$ is a rotation of angle $i\theta$; no $\beta$
- Linear Biases (ALiBi, Press et al. 2022)
  - $\beta_{i, j} = m \cdot(i - j)$ with $m \in \mathbb{R}$


---
### Modern flavors : RMSNorm
- Replaces LayerNorm
- Re-scaling is all you need
$$
RMSNorm_g(a_i) = \frac{a_i}{\sqrt{\frac{1}{N}\sum_{j=1}^N a_j^2}} g_i
$$

---
### Modern flavors : Grouped-Query Attention

<center><img width="1100px" src="../imgs/course3/gqa.png"/></center>

---
<!--_class: lead -->
# Encoder Models

---
### Masked Language Models

<center><img width="1000px" src="../imgs/course3/mlm.svg"/></center>

---
### BERT (Devlin et al., 2018)
- Pre-trained on 128B tokens from Wikipedia + BooksCorpus
- Additional Next Sentence Prediction (NSP) loss
- Two versions:
  - BERT-base (110M parameters)
  - BERT-large (350M parameters)
- **Cost**: ~1000 GPU hours

---
### RoBERTa (Liu et al., 2019)
- Pre-trained on <s>128B</s> **2T** tokens from web data (BERT x10)
- **No more** Next Sentence Prediction (NSP) loss
- Two versions:
  - RoBERTa-base (110M parameters)
  - RoBERTa-large (350M parameters)
- Better results in downstream tasks
- **Cost**: ~25000 GPU hours

---
### Multilingual BERT (mBERT)
- Pre-trained on 128B tokens from multilingual Wikipedia
- 104 languages
- One version:
  - mBERT-base (179M parameters)
- **Cost**: *unknown*

---
### XLM-RoBERTa (Conneau et al., 2019)
- Pre-trained on **63T** tokens from CommonCrawl
- 100 languages
- Two versions:
  - XLM-RoBERTa-base (279M parameters)
  - XLM-RoBERTa-large (561M parameters)
- **Cost**: ~75000 GPU hours

---
### ELECTRA (Clark et al., 2020)
<center><img width="750px" src="../imgs/course3/electra.png"/></center>
<center><img width="550px" src="../imgs/course3/electra_perf.png"/></center>

---
### ELECTRA (Clark et al., 2020)
- Pre-trained on **63T** tokens from CommonCrawl
- 100 languages
- Three versions:
  - ELECTRA-small (14M parameters)
  - ELECTRA-base (110M parameters)
  - ELECTRA-large (350M parameters)
- Really better than BERT/RoBERTa
- **Cost**: =BERT

---
### Encoders: Fine-tuning
<center><img width="750px" src="../imgs/course3/finetuning.svg"/></center>

---
### Encoders: Classical applications
* Natural Language Inference (NLI)
  * *I like cake!* / *Cake is bad* => <s>same</s>|<s>neutral</s>|**opposite**

* Text classification (+ clustering)
  * *I'm so glad to be here!* => joy

* Named Entity Recognition (NER)
  * *I voted for Obama!* => (Obama, pos:3, class:PER)
* and many others...

---
<!--_class: lead -->
# Decoders
---
### Decoders - Motivation

* Models that are designed to **generate text**
* Next-word predictors:
$$
P(w_i\ | (w_j)_{j\neq i}) = P_{\theta}(w_i | w_1...w_{i-1})
$$
* **Problem**: How do we impede self-attention to consider future tokens?

---
### Decoders - Attention mask

<br>
<center><img width="1100px" src="../imgs/course3/attention_mask.svg"/></center>

- Each attention input can only attend to previous positions

---
### Decoders - Causal LM pre-training

- Teacher-forcing
<center><img width="800px" src="../imgs/course3/causal_lm.svg"/></center>

---
### Decoders - Causal LM inference (greedy)

<center><img width="500px" src="../imgs/course3/causal_lm_inference_1.svg"/></center>

---
### Decoders - Causal LM inference (greedy)

<center><img width="500px" src="../imgs/course3/causal_lm_inference_2.svg"/></center>

---
### Decoders - Refining inference

- What we have : a good model for $P_{\theta}(w_i | w_1...w_{i-1})$

- What we want at inference: 
$$
W^* = \argmax_{n, w_i...w_n}P_{\theta}(w_i...w_n | w_1...w_{i-1})
$$

- For a given completion length $n$, there are $|V|^n$ possibilities
  - e.g.: 19 new tokens with a vocab of 30000 tokens > #atoms in $\Omega$
- We need approximations

---
### Decoders - Greedy inference

- Keep best word at each step and start again: 
$$
W^* = \argmax_{n, w_{i+1}...w_n}P_{\theta}(w_{i+1}...w_n | w_1...w_{i-1}w_i^*)
$$
where $w_i^* = \argmax_{w_i} P_{\theta}(w_i | w_1...w_{i-1})$

---
### Decoders - Beam search

- Keep best $k$ chains of tokens at each step:
  - Take $k$ best $w_i$ and compute $P_\theta(w_{i+1} | ...w_i)$ for each
  - Take $k$ best $w_{i+1}$ in each sub-case (now we have $k \times k$ $(w_i, w_{i+1})$ pairs to consider)
  - Consider only the $k$ more likely $(w_i, w_{i+1})$ pairs
  - Compute $P_\theta(w_{i+2} | ...w_iw_{i+1})$ for the $k$ candidates
  - and so on...

---
### Decoders - Top-k sampling

- Randomly sample among top-$k$ tokens based on $P_{\theta}$

<center><img width="500px" src="../imgs/course3/top_k.png"/></center>

---
### Decoders - Top-p (=Nucleus) sampling

- Randomly sample based on $P_{\theta}$ up to $p$%

<center><img width="500px" src="../imgs/course3/top_p.png"/></center>

---
### Decoders - Generation Temperature

- Alter the softmax function:
$$
softmax_\tau(x) = \frac{e^{\frac{x_i}{\tau}}}{\sum_{j}e^{\frac{x_j}{\tau}}}
$$

<center><img width="800px" src="../imgs/course3/temperature.png"/></center>

---
### Decoders - Inference speed
* For greedy decoding without prefix:
  * $n$ passes with sequences of length $1\leq t \leq n$
  * Each pass is $O(n^2)$
  * Complexity: $O(n^3)$
* Other decoding are <ins>more costly</ins>
* Ways to go faster?
---
### Decoders - Query-Key caching

<center><img width="700px" src="../imgs/course3/qk_cache.png"/></center>


---
### Decoders - Speculative decoding

* Generate $\gamma$ tokens using $P_{\phi}$ where $|\phi| \ll |\theta|$ (smaller model)
* Forward $w_i...w_{i+\gamma}$ in teacher-forcing mode and predict $w_{i+\gamma+1}$ with the bigger model
* Compare $P_\theta$ and $P_\phi$ and only keep tokens where they <mark> don't differ too much </mark>

---
<!--_class: lead -->
# Encoder-Decoder models

---
### T5 pre-training

<center><img width="750px" src="../imgs/course3/T5_lm.svg"/></center>

---
### All models can do everything

* Encoders are mostly used to get contextual embeddings
  * They can also generate : $T_{enc}$("I love [MASK]")
* Decoders are mostly used for language generation
  * They can also give contextual embeddings : $T_{dec}$("I love music!")
  * Or solve any task using prompts:
    * "What is the emotion in this tweet? Tweet: '...' Answer:"
* Encoders-decoders are used for language in-filling

---
### Evaluating models

- A useful evaluation metric: ***Perplexity***
- Defined as:
$$
ppl(T_{\theta}; w_1...w_n) = \exp \left( -\frac{1}{n}\sum_{t=1}^{n}\log P_{\theta}(w_t | w_{<t}) \right)
$$

- Other metrics: accuracy, MAUVE, ...

---
### Zero-shot evaluation

* Never-seen problems/data
* Example: *"What is the capital of Italy? Answer:"*
  * Open-ended: Let the model continue the sentence and check exact match
  * Ranking: Get next-word likelihood for *"Rome"*, *"Paris"*, *"London"*, and check if *"Rome"* is best
  * Perplexity: Compute perplexity of *"Rome"* and compare with other models

---
### Few-shot evaluation / In-context learning

* Never-seen problems/data
* Example: *"Paris is the capital of France. London is the capital of the UK. Rome is the capital of"*
* Chain-of-Thought (CoT) examples:
  * Normal: *"(2+3)x5=25. What's (3+4)x2?"*
  * CoT: *"To solve (2+3)x5, we first compute (2+3) = 5 and then multiply (2+3)x5=5x5=25. What's (3+4)x2?"*

---
### Open-sourced evaluation

- Generative models are evaluated on benchmarks
- Example (LLM Leaderboard from HuggingFace):
<center><img width="1200px" src="../imgs/course3/llm_leaderboard.png"/></center>

---
<!--_class: lead -->
# Lab session