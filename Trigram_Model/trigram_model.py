import sys
from collections import defaultdict
import math
import random
import os
import os.path
from collections import deque
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    def add_start_stop_tokens(seq, n):
        if n > 1:
            return ['START'] * (n - 1) + seq + ['STOP']
        else:
            return ['START'] + seq + ['STOP']

    ngrams = []
    if n <= 0:
        return ngrams

    sequence = add_start_stop_tokens(sequence, n)
    
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i + n])
        ngrams.append(ngram)

    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {}
        self.bigramcounts = {}
        self.trigramcounts = {}

        prev_word = "START"
        prev_prev_word = "START"

        for sentence in corpus:
            sentence = ["START"] * 2 + sentence + ["STOP"]
            for word in sentence:
                if word in self.unigramcounts:
                    self.unigramcounts[word] += 1
                else:
                    self.unigramcounts[word] = 1

                if prev_word:
                    bigram = (prev_word, word)
                    if bigram in self.bigramcounts:
                        self.bigramcounts[bigram] += 1
                    else:
                        self.bigramcounts[bigram] = 1
                
                if prev_prev_word:
                    trigram = (prev_prev_word, prev_word, word)
                    if trigram in self.trigramcounts:
                        self.trigramcounts[trigram] += 1
                    else:
                        self.trigramcounts[trigram] = 1
                prev_prev_word = prev_word
                prev_word = word

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram in self.trigramcounts:
            trigram_count = self.trigramcounts[trigram]
            bigram = trigram[:-1]
            if bigram in self.bigramcounts:
                bigram_count = self.bigramcounts[bigram]
                probability = trigram_count / bigram_count
                return probability
        return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram in self.bigramcounts:
            bigram_count = self.bigramcounts[bigram]
            unigram = bigram[-1]
            if unigram in self.unigramcounts:
                unigram_count = self.unigramcounts[unigram]
                probability = bigram_count / unigram_count
                return probability
        return 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        if unigram in self.unigramcounts:
            unigram_count = self.unigramcounts[unigram]
            total_words = sum(self.unigramcounts.values())
            probability = unigram_count / total_words
            return probability
        return 0.0

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        sentence = []
        current_context = ("START", "START")

        for _ in range(t):
            possible_next_words = []
            for word in self.lexicon:
                trigram = current_context + (word,)
                trigram_probability = self.raw_trigram_probability(trigram)
                possible_next_words.extend([word] * int(trigram_probability * 1000))

            next_word = random.choice(possible_next_words)
            sentence.append(next_word)
            current_context = (current_context[-1], next_word)

            if next_word == "STOP":
                break

        return sentence        

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        trigram_prob = self.raw_trigram_probability(trigram)
        bigram = trigram[:-1]
        bigram_prob = self.raw_bigram_probability(bigram)
        unigram = trigram[-1]
        unigram_prob = self.raw_unigram_probability(unigram)

        smoothed_prob = lambda1 * trigram_prob + lambda2 * bigram_prob + lambda3 * unigram_prob

        return smoothed_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        log_prob = 0.0
        trigrams = get_ngrams(sentence, 3)
        for trigram in trigrams:
            if self.smoothed_trigram_probability(trigram) > 0:
                trigram_log_prob = math.log2(self.smoothed_trigram_probability(trigram))
            else:
                trigram_log_prob = float("-inf")

            log_prob += trigram_log_prob
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        total_log_prob = 0.0
        total_tokens = 0
        total_sentences = 0

        for corpus_file in corpus:
            sentences = list(corpus_reader(corpus_file, self.lexicon))

            for sentence in sentences:
                sentence_log_prob = 0.0
                trigrams = get_ngrams(sentence, 3)
                for trigram in trigrams:
                    smoothed_prob = self.smoothed_trigram_probability(trigram)
                    if smoothed_prob > 0:
                        trigram_log_prob = math.log2(smoothed_prob)
                    else:
                        trigram_log_prob = math.log2(1e-10)
                    sentence_log_prob += trigram_log_prob
                
                total_log_prob += sentence_log_prob
                total_tokens += len(sentence)
                total_sentences += 1
        
        # print(total_tokens)
        l = total_log_prob / total_tokens
        perplexity = math.pow(2, -l)

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0

        for f in os.listdir(testdir1):
            pp_model1 = model1.perplexity([os.path.join(testdir1, f)])
            pp_model2 = model2.perplexity([os.path.join(testdir1, f)])
            if pp_model1 < pp_model2:
                correct += 1
            total += 1

        for f in os.listdir(testdir2):
            pp_model2 = model2.perplexity([os.path.join(testdir2, f)])
            pp_model1 = model1.perplexity([os.path.join(testdir2, f)])
            if pp_model1 > pp_model2:
                correct += 1
            total += 1

        return correct / total

if __name__ == "__main__":
    model = TrigramModel("brown_train.txt")

    # part1 test
    print(get_ngrams(["natural", "language", "processing"], 1))
    print(get_ngrams(["natural", "language", "processing"], 2))
    print(get_ngrams(["natural", "language", "processing"], 3))

    # part2 test
    print(model.trigramcounts[('START','START','the')])
    print(model.bigramcounts[('START','the')])
    print(model.unigramcounts[('the')])

    # part3 test
    print(model.raw_trigram_probability(('START','START','the')))
    print(model.raw_bigram_probability(('START','the')))
    print(model.raw_unigram_probability(('the')))

    print(model.generate_sentence())

    # part4 test
    print(model.smoothed_trigram_probability(('START','START','the')))

    # part5 test
    print(model.sentence_logprob(['the', 'fulton', 'county', 'grand', 'jury', 'said', 'friday', 'an', 'investigation', 'of', 'atlanta', "'s", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.']))

    # part6 test
    train_corpus = ["brown_train.txt"]
    pp = model.perplexity(train_corpus)
    print("Training Perplexity of the model is: ", pp)

    dev_corpus = ["brown_test.txt"]
    pp = model.perplexity(dev_corpus)
    print("Testing Perplexity of the model is: ", pp)

    # past7 test
    acc = essay_scoring_experiment('train_high.txt', "train_low.txt", "test_high", "test_low")
    print("The accuracy of the model:", acc)

