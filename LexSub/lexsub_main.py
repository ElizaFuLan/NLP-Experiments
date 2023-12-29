#!/usr/bin/env python
import string
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow_hub as hub

from sentence_transformers import SentenceTransformer

import gensim
import transformers 

from typing import List

from nltk.stem import WordNetLemmatizer

import gensim.downloader as api
# glove_model = api.load("glove-twitter-200")
glove_model = api.load("glove-wiki-gigaword-300")
from scipy.spatial.distance import cosine

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    pos_map = {'a': wn.ADJ, 'n': wn.NOUN, 'v': wn.VERB, 'r': wn.ADV}
    pos_wn = pos_map.get(pos)

    synsets = wn.synsets(lemma, pos=pos_wn)

    candidates = set()
    for synset in synsets:
        for lemma_syn in synset.lemmas():
            if lemma_syn.name() != lemma:
                candidate = lemma_syn.name().replace('_', ' ')
                candidates.add(candidate)

    return list(candidates)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context): #replace for part 2
    lemma = context.lemma
    pos = context.pos

    candidates = get_candidates(lemma, pos)
    max_freq = 0
    best_candidate = None

    for candidate in candidates:
        total_freq = 0
        for synset in wn.synsets(candidate, pos=pos):
            if lemma in [lem.name() for lem in synset.lemmas()]:
                total_freq += sum(lem.count() for lem in synset.lemmas() if lem.name() == candidate)

        if total_freq > max_freq:
            max_freq = total_freq
            best_candidate = candidate

    return best_candidate if best_candidate else lemma

def wn_simple_lesk_predictor(context): #replace for part 3
    stop_words = stopwords.words('english')
    lemma = context.lemma
    pos = context.pos

    best_candidate = None
    max_score = -1

    for synset in wn.synsets(lemma, pos=pos):
        definition = tokenize(synset.definition())
        examples = tokenize(' '.join(synset.examples()))

        overlap = len(set(definition + examples) & set(context.left_context + context.right_context) - set(stop_words))
        
        for lem in synset.lemmas():
            if lem.name() == lemma:
                continue

            score = 1000 * overlap + 100 * synset.lemmas()[0].count() + lem.count()

            if score > max_score:
                max_score = score
                best_candidate = lem

    return best_candidate.name().replace('_', ' ') if best_candidate else lemma 
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str: # replace for part4
        synonyms = get_candidates(context.lemma, context.pos)

        best_synonym = None
        max_similarity = -1

        if context.lemma in self.model:
            for synonym in synonyms:
                if synonym in self.model:
                    similarity = self.model.similarity(context.lemma, synonym)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_synonym = synonym

        return best_synonym if best_synonym else 'smurf'


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str: # part5
        candidates = get_candidates(context.lemma, context.pos)

        masked_sentence = context.left_context + ['[MASK]'] + context.right_context
        masked_sentence = " ".join(masked_sentence)

        input_ids = self.tokenizer.encode(masked_sentence, return_tensors="tf")

        outputs = self.model(input_ids)
        predictions = outputs[0]

        mask_index = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        mask_position = np.where(input_ids == mask_index)[1][0]

        best_word = ''
        highest_score = -np.inf
        for candidate in candidates:
            candidate_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(candidate))
            if len(candidate_id) == 1:
                score = predictions[0, mask_position, candidate_id[0]].numpy()
                if score > highest_score:
                    best_word = candidate
                    highest_score = score

        return best_word if best_word else 'smurf'
    
# part 6
# In this part, I used GloVe model and combined it with the BERT model. 
# In the prediction part, I calculated each grade on GloVe and BERT, and combined them together as a final judging grade.
# Finally, it showed a better evaluation grade than professional BERT model.
# The grade was shown as followed:
# Total = 298, attempted = 298
# precision = 0.129, recall = 0.129
# Total with mode 206 attempted 206
# precision = 0.199, recall = 0.199
class SelfPredictor(object):
    def __init__(self, glove_model): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.glove_model = glove_model

    def predict(self, context: Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)

        masked_sentence = context.left_context + ['[MASK]'] + context.right_context
        masked_sentence = " ".join(masked_sentence)
        input_ids = self.tokenizer.encode(masked_sentence, return_tensors="tf")
        outputs = self.model(input_ids)
        predictions = outputs[0]
        mask_index = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        mask_position = np.where(input_ids == mask_index)[1][0]

        best_word = ''
        highest_combined_score = -np.inf
        for candidate in candidates:
            candidate_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(candidate))
            if len(candidate_id) == 1:
                bert_score = predictions[0, mask_position, candidate_id[0]].numpy()
                
                if candidate in self.glove_model and context.lemma in self.glove_model:
                    glove_similarity = 1 - cosine(self.glove_model[candidate], self.glove_model[context.lemma])
                else:
                    glove_similarity = 0

                combined_score = bert_score + 20 * glove_similarity

                if combined_score > highest_combined_score:
                    highest_combined_score = combined_score
                    best_word = candidate

        return best_word if best_word else 'smurf'


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # bert_predictor = BertPredictor()

    self_predictor = SelfPredictor(glove_model)

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
         
        # part 0
        # prediction = smurf_predictor(context) 
        # print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
        
        # part 2
        # prediction = wn_frequency_predictor(context)
        # print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

        # part 3
        # prediction = wn_simple_lesk_predictor(context)
        # print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

        # part 4
        # prediction = predictor.predict_nearest(context)
        # print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

        # part 5
        # prediction = bert_predictor.predict(context)
        # print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

        # part 6
        prediction = self_predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))