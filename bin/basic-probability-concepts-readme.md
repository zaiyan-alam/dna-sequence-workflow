# Chapter: Basic Probability Concepts

Probability theory is fundamental to Natural Language Processing (NLP) as it underpins many models and algorithms used to interpret, generate, and understand human language. This chapter introduces the foundational concepts of probability, focusing on **Sample Spaces and Events** and the **Probability Axioms**. We'll explore these topics with NLP-related mathematical examples, explain the notations and expressions, and use tables for visualization where applicable.

---

## 1. Sample Spaces and Events

### 1.1. Sample Space

**Definition:**  
A **sample space** (denoted as $\Omega$) is the set of all possible outcomes of a random experiment.

**In NLP Context:**  
Consider the task of predicting the next word in a sentence. The sample space consists of all possible words that could follow the given context.

**Example:**  
Suppose we are using a unigram model to predict the next word in the sentence "The cat sat on the ___". The sample space $\Omega$ could be:

$$\Omega = \{\text{"mat"}, \text{"chair"}, \text{"floor"}, \text{"bed"}, \dots \}$$

Each element in $\Omega$ represents a possible next word.

### 1.2. Events

**Definition:**  
An **event** is a subset of the sample space. It represents one or more outcomes that share a common property.

**Notations:**
- $A, B, C, \dots$ denote events.
- $\mathbb{P}(A)$ denotes the probability of event $A$.

**In NLP Context:**  
An event could represent the occurrence of a specific word or a set of words.

**Example 1: Specific Word Event**  
Let $A$ be the event that the next word is "mat":

$$A = \{\text{"mat"}\}$$

**Example 2: Set of Words Event**  
Let $B$ be the event that the next word is either "mat" or "chair":

$$B = \{\text{"mat"}, \text{"chair"}\}$$

### 1.3. Visualization with Tables

Let's visualize a simplified sample space and some events related to predicting the next word.

| Outcome Index | Possible Next Word |
|---------------|--------------------|
| 1             | mat                |
| 2             | chair              |
| 3             | floor              |
| 4             | bed                |
| 5             | table              |

**Sample Space ($\Omega$)**:

$$\Omega = \{ \text{"mat"}, \text{"chair"}, \text{"floor"}, \text{"bed"}, \text{"table"} \}$$

**Events:**

| Event | Description                           | Outcomes                |
|-------|---------------------------------------|-------------------------|
| $A$ | Next word is "mat"                  | $\{ \text{"mat"} \}$             |
| $B$ | Next word is "mat" or "chair"       | $\{ \text{"mat"}, \text{"chair"} \}$    |
| $C$ | Next word is a furniture item       | $\{ \text{"chair"}, \text{"bed"}, \text{"table"} \}$ |
| $D$ | Next word starts with 'b' or 'm'    | $\{ \text{"mat"}, \text{"bed"} \}$      |

### 1.4. Mathematical Example in NLP

**Problem:**  
Given the sample space $\Omega = \{ \text{"mat"}, \text{"chair"}, \text{"floor"}, \text{"bed"}, \text{"table"} \}$, suppose the probabilities of each word occurring next are as follows:

| Word   | Probability ($\mathbb{P}$) |
|--------|---------------------------|
| mat    | 0.2                       |
| chair  | 0.3                       |
| floor  | 0.1                       |
| bed    | 0.25                      |
| table  | 0.15                      |

**Question:**  
What is $\mathbb{P}(B)$, the probability that the next word is either "mat" or "chair"?

**Solution:**  
Event $B = \{ \text{"mat"}, \text{"chair"} \}$

Using the **Addition Rule** for mutually exclusive events:

$$\mathbb{P}(B) = \mathbb{P}(\text{"mat"}) + \mathbb{P}(\text{"chair"}) = 0.2 + 0.3 = 0.5$$

---

## 2. Probability Axioms

Probability axioms are the foundational rules that any probability measure must satisfy. These axioms ensure consistency and logical coherence in probabilistic reasoning.

### 2.1. Axiom 1: Non-negativity

**Statement:**  
For any event $A$, the probability is non-negative:

$$\mathbb{P}(A) \geq 0$$

**Explanation:**  
Probabilities cannot be negative; they represent the likelihood of events occurring.

**NLP Example:**  
The probability of any word or set of words occurring next in a sentence must be zero or positive. For instance:

$$\mathbb{P}(\text{"mat"}) = 0.2 \geq 0$$

### 2.2. Axiom 2: Normalization

**Statement:**  
The probability of the entire sample space is 1:

$$\mathbb{P}(\Omega) = 1$$

**Explanation:**  
This axiom ensures that some outcome within the sample space must occur.

**NLP Example:**  
The sum of probabilities of all possible next words must equal 1.

Using our previous example:

$$\mathbb{P}(\text{"mat"}) + \mathbb{P}(\text{"chair"}) + \mathbb{P}(\text{"floor"}) + \mathbb{P}(\text{"bed"}) + \mathbb{P}(\text{"table"}) = 0.2 + 0.3 + 0.1 + 0.25 + 0.15 = 1.0$$

### 2.3. Axiom 3: Additivity

**Statement:**  
For any countable sequence of mutually exclusive events $A_1, A_2, A_3, \dots$:

$$\mathbb{P}\left( \bigcup_{i=1}^{\infty} A_i \right) = \sum_{i=1}^{\infty} \mathbb{P}(A_i)$$

**Explanation:**  
If events cannot occur simultaneously (mutually exclusive), the probability of any one of them occurring is the sum of their individual probabilities.

**NLP Example:**  
Events $A$ and $B$ are mutually exclusive if they share no common outcomes. For instance:

- $A = \{\text{"mat"}\}$
- $B = \{\text{"chair"}\}$

Since "mat" and "chair" are distinct words, $A$ and $B$ are mutually exclusive.

Thus,

$$\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) = 0.2 + 0.3 = 0.5$$

### 2.4. Visualization with Tables

Let's verify the probability axioms using our sample space.

| Axiom         | Description                                             | Verification with Example        |
|---------------|---------------------------------------------------------|-----------------------------------|
| Non-negativity | $\mathbb{P}(A) \geq 0$ for all $A$            | All word probabilities $\geq 0$ |
| Normalization   | $\mathbb{P}(\Omega) = 1$                          | $0.2 + 0.3 + 0.1 + 0.25 + 0.15 = 1.0$ |
| Additivity      | $\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B)$ if $A \cap B = \emptyset$ | $\mathbb{P}(\{\text{"mat"}\} \cup \{\text{"chair"}\}) = 0.2 + 0.3 = 0.5$ |

### 2.5. Mathematical Example in NLP

**Problem:**  
Assume you have the following probabilities for bigrams (pairs of words) in a corpus:

| Bigram           | Probability ($\mathbb{P}$) |
|------------------|---------------------------|
| ("the", "cat")   | 0.05                      |
| ("the", "dog")   | 0.04                      |
| ("the", "mouse") | 0.01                      |
| ("a", "cat")     | 0.03                      |
| ("a", "dog")     | 0.02                      |
| ("a", "mouse")   | 0.01                      |
| **Total**        | **0.16**                  |

**Question 1:**  
Verify if the given probabilities satisfy the **Normalization Axiom** for the sample space of bigrams starting with "the" or "a".

**Solution:**  
Sample space $\Omega = \{ \text{("the", "cat")}, \text{("the", "dog")}, \text{("the", "mouse")}, \text{("a", "cat")}, \text{("a", "dog")}, \text{("a", "mouse")} \}$

Sum of probabilities:

$$0.05 + 0.04 + 0.01 + 0.03 + 0.02 + 0.01 = 0.16$$

**Issue:**  
The sum is $0.16 \neq 1$. Therefore, the probabilities do **not** satisfy the Normalization Axiom.

**Solution:**  
To satisfy the axiom, probabilities should be normalized. Divide each probability by the total sum $0.16$:

| Bigram           | Normalized Probability ($\mathbb{P}'$) |
|------------------|----------------------------------------|
| ("the", "cat")   | $0.05 / 0.16 \approx 0.3125$           |
| ("the", "dog")   | $0.04 / 0.16 = 0.25$                   |
| ("the", "mouse") | $0.01 / 0.16 \approx 0.0625$           |
| ("a", "cat")     | $0.03 / 0.16 \approx 0.1875$           |
| ("a", "dog")     | $0.02 / 0.16 = 0.125$                  |
| ("a", "mouse")   | $0.01 / 0.16 \approx 0.0625$           |
| **Total**        | **1.0**                                |

**Question 2:**  
Given the normalized probabilities, verify the **Additivity Axiom** for events $A = \{\text{("the", "cat")}, \text{("the", "dog")}\}$ and $B = \{\text{("a", "mouse")}\}$.

**Solution:**

- $\mathbb{P}(A) = 0.3125 + 0.25 = 0.5625$
- $\mathbb{P}(B) = 0.0625$
- $\mathbb{P}(A \cup B) = 0.3125 + 0.25 + 0.0625 = 0.625$

Since $A$ and $B$ are mutually exclusive:

$$\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) = 0.5625 + 0.0625 = 0.625$$

**Verification:**  
Additivity holds as both sides equal $0.625$.

---

## 3. Additional Important Information

### 3.1. Mutually Exclusive Events

**Definition:**  
Events $A$ and $B$ are **mutually exclusive** (disjoint) if they cannot occur simultaneously, i.e., $A \cap B = \emptyset$.

**NLP Example:**  
In word prediction, the event that the next word is "mat" and "chair" simultaneously is impossible, making these events mutually exclusive.

### 3.2. Complementary Events

**Definition:**  
The **complement** of an event $A$, denoted $A^c$, consists of all outcomes in the sample space that are not in $A$.

$$\mathbb{P}(A^c) = 1 - \mathbb{P}(A)$$

**NLP Example:**  
If $A = \{\text{"mat"}\}$, then $A^c$ includes all other words in $\Omega$:

$$A^c = \{\text{"chair"}, \text{"floor"}, \text{"bed"}, \text{"table"}\}$$

### 3.3. Union and Intersection of Events

- **Union ($A \cup B$)**: Event that either $A$ or $B$ occurs.
- **Intersection ($A \cap B$)**: Event that both $A$ and $B$ occur simultaneously.

**NLP Example:**

- **Union:** The event that the next word is "mat" or "chair".
- **Intersection:** If $A = \{\text{"mat"}, \text{"chair"}\}$ and $B = \{\text{"chair"}, \text{"table"}\}$, then $A \cap B = \{\text{"chair"}\}$.

### 3.4. Conditional Probability (Brief Introduction)

While not covered in depth in this chapter, **conditional probability** is essential in NLP, especially in models like Hidden Markov Models (HMMs) and Conditional Random Fields (CRFs).

**Definition:**  
The probability of event $A$ given that event $B$ has occurred is denoted as $\mathbb{P}(A|B)$ and is defined as:

$$\mathbb{P}(A|B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}$$

**NLP Example:**  
In a bigram model, the probability of the word "dog" given the previous word "the" is $\mathbb{P}(\text{"dog"}|\text{"the"})$.

---

## 4. Python Implementation Examples

Understanding probability concepts is further reinforced through practical implementation. Below are simple Python examples illustrating sample spaces, events, and probability axioms.

### 4.1. Defining Sample Space and Events

```python
# Define the sample space
sample_space = ["mat", "chair", "floor", "bed", "table"]

# Define events
A = {"mat"}
B = {"mat", "chair"}
C = {"chair", "bed", "table"}
D = {"mat", "bed"}

# Define probabilities
probabilities = {
    "mat": 0.2,
    "chair": 0.3,
    "floor": 0.1,
    "bed": 0.25,
    "table": 0.15
}
```

### 4.2. Verifying Normalization Axiom

```python
# Sum of probabilities
total_prob = sum(probabilities.values())
print(f"Total Probability: {total_prob}")  # Output should be 1.0

# If not normalized, normalize the probabilities
if total_prob != 1.0:
    probabilities = {word: prob / total_prob for word, prob in probabilities.items()}
```

### 4.3. Calculating Probability of an Event

```python
def probability(event, probabilities):
    return sum(probabilities[word] for word in event)

# Probability of event B
prob_B = probability(B, probabilities)
print(f"P(B) = {prob_B}")  # Output: 0.5
```

### 4.4. Checking Additivity Axiom

```python
# Define mutually exclusive events
A = {"mat"}
B = {"chair"}

# P(A ∪ B)
P_A_union_B = probability(A.union(B), probabilities)

# P(A) + P(B)
P_A_plus_P_B = probability(A, probabilities) + probability(B, probabilities)

print(