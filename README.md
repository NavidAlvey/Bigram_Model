# Bigram Language Model

This project implements a bigram language model using the Brown corpus from the Natural Language Toolkit (NLTK). 
It utilizes Laplace smoothing to estimate bigram probabilities, allowing it to compute sentence probabilities and predict likely next words in a sequence.

## Features
- **Trainable Bigram Model**: Builds unigram and bigram counts from text data.
- **Sentence Probability Calculation**: Computes the probability of a sentence using bigram probabilities with smoothing.
- **Next Word Prediction**: Suggests the most likely next words for a given prefix.

## Requirements
- Python 3.x
- NLTK library (`pip install nltk`)
- Download the Brown corpus in NLTK:
  ```python
  import nltk
  nltk.download('brown')

## Usage
### Initialize and Train the Model
```python
bigram_model = BigramModel()
bigram_model.train(brown_sents)
```
### Calculate Sentence Probability
```python
sentence = "The dog barked at the cat"
probability = bigram_model.sentence_probability(sentence)
print(f"Sentence probability: {probability}")
```
### Predict Next Words
```python
prefix = "I won 200"
next_words = bigram_model.predict_next_word(prefix, top_n=5)
print(f"Predicted next words: {next_words}")
```



