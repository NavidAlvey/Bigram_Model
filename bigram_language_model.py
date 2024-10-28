import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import math

# Load Brown Corpus
brown_words = brown.words()

# Add start and end tokens
brown_sents = brown.sents()
brown_sents = [['<s>'] + sent + ['</s>'] for sent in brown_sents]

# Build bigram model with Laplace smoothing
class BigramModel:
    def __init__(self):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.vocab = set()
        self.total_words = 0
    
    def train(self, sentences):
        for sentence in sentences:
            for i in range(len(sentence) - 1):
                word1 = sentence[i].lower()
                word2 = sentence[i + 1].lower()
                self.unigram_counts[word1] += 1
                self.bigram_counts[(word1, word2)] += 1
                self.vocab.update([word1, word2])
            self.unigram_counts[sentence[-1].lower()] += 1
        self.total_words = sum(self.unigram_counts.values())
    
    def bigram_probability(self, word1, word2, alpha=1):
        word1 = word1.lower()
        word2 = word2.lower()
        vocab_size = len(self.vocab)
        bigram_count = self.bigram_counts[(word1, word2)]
        unigram_count = self.unigram_counts[word1]
        # Laplace smoothed probability
        return (bigram_count + alpha) / (unigram_count + alpha * vocab_size)
    
    def sentence_probability(self, sentence):
        sentence = ['<s>'] + sentence.lower().split() + ['</s>']
        log_prob = 0
        for i in range(len(sentence) - 1):
            word1 = sentence[i]
            word2 = sentence[i + 1]
            prob = self.bigram_probability(word1, word2)
            log_prob += math.log(prob)  # Summation of all log probabilities to avoid overflow
        return math.exp(log_prob)
    
    def predict_next_word(self, prefix, top_n=5):
        last_word = prefix.lower().split()[-1]
        candidates = []
        for word in self.vocab:
            prob = self.bigram_probability(last_word, word)
            candidates.append((word, prob))
        # Sort by probability and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in candidates[:top_n]]

# Initialize and train model
bigram_model = BigramModel()
bigram_model.train(brown_sents)

sentence = "The dog barked at the cat"
probability = bigram_model.sentence_probability(sentence)
print(f"Sentence probability with smoothing: \n'{sentence}': {probability}")

prefix = "I won 200"
next_words = bigram_model.predict_next_word(prefix, top_n=5)
print(f"Predicted next 5 words: \n{next_words}")
