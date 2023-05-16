---
theme: gaia
_class: lead
paginate: true
title: "Course 2: Tokenization"
backgroundColor: #fff
marp: true
---

<style scoped>
/* Reset table styling provided by theme */
table, tr, td, th {
  all: unset;

  /* Override contextual styling */
  border: 0 !important;
  background: transparent !important;
}
table { display: table; }
tr { display: table-row; }
td, th { display: table-cell; }

/* ...and layout freely :) */
table {
  width: 100%;
}
td {
  text-align: center;
  vertical-align: middle;
}
</style>

# **Course 2: Tokenization**



---
<!--footer: 'Course 2: Tokenization' -->

### What is tokenization?

Turning text...
```markdown
I love playing soccer!
```

...into *tokens*
```markdown
['I', 'love', 'play', 'ing', 'soccer', '!']
```

---
<!--_class: lead -->
# Historical Notions

---

### *Tokenization Origins*

The concept comes from linguistics
> *non-empty contiguous sequence of graphemes or phonemes in a document*
$\approx$
split on blanks


---

### *Tokenization Origins*
<br>
<br>


```markdown
old_tokenize("I love playing soccer!") = ['I', 'love', 'playing', 'soccer!']
```

- Different from *word-forms* :warning:
    - *damélo*  &rarr;  *da*/*mé*/*lo* (=*give*/*me*/*it*)

---

### *Tokenization Origins*

Natural language is split into...
<br>

- Sentences, utterances, documents... (*macroscopical*)
that are split into...<br>

    - Tokens, word-forms... (*microscopical*)

&rarr; Used for linguistic tasks (POS tagging, syntax parsing,...)

---

### Tokenization & ML

Machine Learning relies on tokenization:
-  Gives better performance
-  **Fixed-size vocabulary** often required

---

### Tokenization & ML

Evolution of modeling complexity w.r.t. the sequence length
<center><table>
<tr><td><b>Model Type</b></td><td><b>Year</b></td><td><b>Complexity</b></td></tr>
<tr><td>Tf-Idf</td><td>1972</td><td>O(1)</td></tr>
<tr><td>RNNs</td><td>~1985</td><td>O(n)</td></tr>
<tr><td>Transformers</td><td>2017</td><td>O(n^2)</td></tr>
</table></center>

&rarr; Need for a good downsampling mechanism 

---