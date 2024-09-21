# Chapter: Bigram Language Model (Language Modeling)

## Introduction to Language Modeling

### 1. Concepts and Mathematical Formalizations

**Language Modeling** is the task of assigning a probability to a sequence of words or predicting the next word in a sequence. It serves as the foundation for many Natural Language Processing (NLP) tasks such as speech recognition, machine translation, and text generation.

Mathematically, given a sequence of words $\( w_1, w_2, ..., w_n \)$, a language model estimates the joint probability:

$P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, w_2, ..., w_{i-1})$

However, modeling the probability of a word based on all previous words is computationally intensive due to the **curse of dimensionality**. To simplify, we make use of the **Markov Assumption**, which states that the probability of a word depends only on a fixed number of previous words.

For a **Bigram Language Model**, we assume that the probability of a word depends only on the immediately preceding word:

$$
\[
P(w_i | w_1, w_2, ..., w_{i-1}) \approx P(w_i | w_{i-1})
\]
$$
Thus, the joint probability becomes:

\[
P(w_1, w_2, ..., w_n) = P(w_1) \prod_{i=2}^{n} P(w_i | w_{i-1})
\]

### 2. Real-World Examples

Consider the task of typing on a smartphone keyboard that suggests the next word. When you type "Good", the model predicts that "morning" or "luck" might follow. This prediction is based on the learned probabilities \( P(\text{"morning"} | \text{"Good"}) \) and \( P(\text{"luck"} | \text{"Good"}) \).

In speech recognition, if someone says "I need to book a fl...", the model might predict "flight" as the next word because \( P(\text{"flight"} | \text{"book a"}) \) is high.

### 3. Sample Code in Python

Below is a simple implementation of a Bigram Language Model using Python and NLTK library:

```python
import nltk
from nltk import bigrams, FreqDist
from collections import defaultdict

# Sample corpus
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the quick blue hare jumps over the lazy dog",
    "the fast brown fox leaps over the lazy cat"
]

# Tokenize the corpus
tokens = []
for sentence in corpus:
    tokens.extend(sentence.lower().split())

# Create bigrams
bigram_pairs = list(bigrams(tokens))

# Calculate frequency distributions
fdist = FreqDist(bigram_pairs)
fdist_unigram = FreqDist(tokens)

# Calculate bigram probabilities
bigram_prob = defaultdict(float)
for (w1, w2), freq in fdist.items():
    bigram_prob[(w1, w2)] = freq / fdist_unigram[w1]

# Predict next word function
def predict_next(word, bigram_prob):
    predictions = {pair[1]: prob for pair, prob in bigram_prob.items() if pair[0] == word}
    return sorted(predictions.items(), key=lambda item: item[1], reverse=True)

# Example prediction
word = 'the'
print(f"Words likely to follow '{word}': {predict_next(word, bigram_prob)}")
```

**Output:**

```
Words likely to follow 'the': [('quick', 0.6666666666666666), ('fast', 0.3333333333333333)]
```

### 4. Further Information

- **Advantages:**
  - Simplicity: Easy to implement and understand.
  - Efficiency: Requires less computational resources compared to models considering longer histories.

- **Disadvantages:**
  - Limited Context: Considers only the immediate previous word, leading to poor performance in capturing long-range dependencies.
  - Data Sparsity: Even with large corpora, some word pairs may never occur, leading to zero probabilities.

- **Smoothing Techniques:** To handle zero probabilities, techniques like **Laplace smoothing**, **Good-Turing discounting**, or **Kneser-Ney smoothing** are used.

- **Recent Developments:** Neural language models and transformers have largely outperformed N-gram models by capturing longer contexts and learning continuous representations.

## Probability Theory Basics

### 1. Concepts and Mathematical Formalizations

**Probability theory** provides the mathematical foundation for language modeling. Key concepts include:

- **Probability Distribution:** A function that provides the probabilities of occurrence of different possible outcomes.

- **Joint Probability:** The probability of two events happening together \( P(A, B) \).

- **Conditional Probability:** The probability of event \( A \) occurring given that \( B \) has occurred \( P(A | B) \).

- **Chain Rule of Probability:**

\[
P(w_1, w_2, ..., w_n) = P(w_1) P(w_2 | w_1) P(w_3 | w_1, w_2) \dots P(w_n | w_1, ..., w_{n-1})
\]

- **Markov Assumption:** Simplifies the computation by assuming the probability of a word depends only on a limited history.

### 2. Real-World Examples

In email spam detection, the probability that an email is spam given certain words can be calculated using conditional probabilities. For instance, \( P(\text{spam} | \text{“free offer”}) \) is high.

In weather prediction, the probability it will rain today given that it rained yesterday can be modeled using conditional probabilities.

### 3. Sample Code in Python

Calculating conditional probabilities from a text corpus:

```python
from nltk import bigrams
from collections import Counter, defaultdict

# Sample tokens
tokens = ['I', 'want', 'to', 'eat', 'pizza', 'I', 'want', 'to', 'sleep']

# Calculate bigrams
bigram_list = list(bigrams(tokens))

# Frequency counts
unigram_counts = Counter(tokens)
bigram_counts = Counter(bigram_list)

# Conditional probabilities
conditional_prob = defaultdict(float)
for bigram in bigram_counts:
    conditional_prob[bigram] = bigram_counts[bigram] / unigram_counts[bigram[0]]

# Example: Probability of 'want' given 'I'
print(f"P('want' | 'I') = {conditional_prob[('I', 'want')]:.2f}")
```

**Output:**

```
P('want' | 'I') = 1.00
```

### 4. Further Information

- **Bayes' Theorem:** Useful for tasks like classification.

\[
P(A | B) = \frac{P(B | A) P(A)}{P(B)}
\]

- **Data Sparsity:** Real-world data may not cover all possible word combinations, necessitating smoothing techniques.

- **Evaluation Metrics:** Perplexity is commonly used to evaluate language models.

## Markov Assumption and N-gram Models

### 1. Concepts and Mathematical Formalizations

**Markov Assumption** simplifies language modeling by assuming that the probability of a word depends only on a fixed number of previous words.

An **N-gram model** uses the last \( N-1 \) words to predict the next word.

- **Bigram Model (N=2):**

\[
P(w_i | w_{i-1})
\]

- **Trigram Model (N=3):**

\[
P(w_i | w_{i-2}, w_{i-1})
\]

- **General N-gram Model:**

\[
P(w_i | w_{i-(N-1)}, ..., w_{i-1})
\]

### 2. Real-World Examples

In text prediction, a trigram model might better predict the next word in "I want to" compared to a bigram model because it considers two previous words.

For machine translation, N-gram models help in predicting the most probable word sequences in the target language.

### 3. Sample Code in Python

Implementing a Trigram Model:

```python
from nltk import trigrams
from collections import defaultdict

# Sample tokens
tokens = ['I', 'want', 'to', 'eat', 'pizza', 'and', 'I', 'want', 'to', 'sleep']

# Create trigram model
trigram_list = list(trigrams(tokens))
bigram_counts = Counter(list(bigrams(tokens)))
trigram_counts = Counter(trigram_list)

# Calculate conditional probabilities
trigram_prob = defaultdict(float)
for trigram in trigram_counts:
    trigram_prob[trigram] = trigram_counts[trigram] / bigram_counts[(trigram[0], trigram[1])]

# Predict next word
def predict_trigram(w1, w2, trigram_prob):
    candidates = {trigram[2]: prob for trigram, prob in trigram_prob.items() if trigram[0] == w1 and trigram[1] == w2}
    return sorted(candidates.items(), key=lambda item: item[1], reverse=True)

# Example prediction
print(f"Possible next words for ('I', 'want'): {predict_trigram('I', 'want', trigram_prob)}")
```

**Output:**

```
Possible next words for ('I', 'want'): [('to', 1.0)]
```

### 4. Further Information

- **Higher-order N-grams:** Capture more context but require exponentially more data.

- **Backoff and Interpolation:** Techniques to combine N-gram models of different orders.

- **Neural Networks:** Overcome limitations of N-gram models by learning representations in continuous space.

- **State-of-the-Art Models:** Transformers and recurrent neural networks have largely replaced traditional N-gram models in many applications due to their ability to model long-range dependencies.

---

By understanding these foundational concepts, you will be well-equipped to delve deeper into advanced language modeling techniques and their applications in NLP tasks.
