import math, collections
import itertools
import re
import pandas as pd
import nltk
nltk.download('punkt')
import nltk.data
import enchant
from lingua import Language, LanguageDetectorBuilder
#from easynmt import EasyNMT
from googletrans import Translator
import sys
import gc
import datetime

# """
#   Define the sentence correction class
# """
class Sentence_Corrector:
    def __init__(self, training_file):
        sys.setrecursionlimit(10**6)
        self.laplaceUnigramCounts = collections.defaultdict(lambda: 0)
        self.laplaceBigramCounts = collections.defaultdict(lambda: 0)
        self.laplaceSpanishUnigramCounts = collections.defaultdict(lambda: 0)
        self.laplaceSpanishBigramCounts = collections.defaultdict(lambda: 0)
        self.total = 0
        self.spanish_total = 0
        self.sentences = []
        self.search_weight = 2
        self.search_dict = {}
        self.brand_weight = 10
        self.brand_dict = {}
        self.importantKeywords = set()
        self.dict = enchant.PyPWL('custom_dict.txt')
        self.spanish_dict = enchant.PyPWL('spanish_dict.txt')
        self.spanish_freq_dict = {}
        self.tokenize_file(training_file)
        self.dataframe_to_dict('./top_search.xlsx', './brand_freq.csv', './spanish_freq.csv')
        self.train_english()
        self.train_spanish()
        self.is_spanish = False
        del self.spanish_freq_dict
        del self.search_dict
        del self.brand_dict
        del self.sentences
        gc.collect()

    def tokenize_file(self, file):
        # """
        #   Read the file, tokenize and build a list of sentences
        # """
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        f = open(file)
        content = f.read()
        for sentence in tokenizer.tokenize(content):
            sentence_clean = [i.lower() for i in re.split('[^a-zA-Z0-9%]+', sentence) if i]
            self.sentences.append(sentence_clean)

    def dataframe_to_dict(self, search_df_name, brand_df_name, spanish_df_name):
        search_df = pd.read_excel(search_df_name)
        self.search_dict = pd.Series(search_df.frequencies.values, index=search_df.search_term_ignore_case).to_dict()
        brand_df = pd.read_csv(brand_df_name)
        self.brand_dict = pd.Series(brand_df.quantity.values, index=brand_df.brand).to_dict()
        spanish_df = pd.read_csv(spanish_df_name)
        self.spanish_freq_dict = pd.Series(spanish_df.frequencies.values, index=spanish_df.spanish).to_dict()

    def train_spanish(self):
        # """
        #   Train unigram and bigram
        # """
        # assume search_dict looks something like this: {'apple watch': 100, 'orange': 100}
        for key in self.spanish_freq_dict:
            temp = [i.lower() for i in re.split(" ", str(key)) if i]
            temp.insert(0, '<s>')
            temp.append('</s>')
            for i in range(len(temp) - 1):
                token1 = temp[i]
                token2 = temp[i + 1]
                count = self.spanish_freq_dict[key]
                self.laplaceSpanishUnigramCounts[token1] += count
                self.laplaceSpanishBigramCounts[(token1, token2)] += count
                self.spanish_total += count
            self.spanish_total += count
            self.laplaceSpanishUnigramCounts[temp[-1]] += count

    def train_english(self):
        # """
        #   Train unigram and bigram
        # """
        for sentence in self.sentences:
            sentence.insert(0, '<s>')
            sentence.append('</s>')
            for i in range(len(sentence) - 1):
                token1 = sentence[i]
                token2 = sentence[i + 1]
                self.laplaceUnigramCounts[token1] += 1
                self.laplaceBigramCounts[(token1, token2)] += 1
                self.total += 1
            self.total += 1
            self.laplaceUnigramCounts[sentence[-1]] += 1

        # assume search_dict looks something like this: {'apple watch': 100, 'orange': 100}
        for key in self.search_dict:
            temp = [i.lower() for i in re.split('[^a-zA-Z0-9%]+', str(key)) if i]
            temp.insert(0, '<s>')
            temp.append('</s>')
            for i in range(len(temp) - 1):
                token1 = temp[i]
                token2 = temp[i + 1]
                count = self.search_dict[key] * self.search_weight
                self.laplaceUnigramCounts[token1] += count
                self.laplaceBigramCounts[(token1, token2)] += count
                self.total += count
            self.total += count
            self.laplaceUnigramCounts[temp[-1]] += count

        for key in self.brand_dict:
            temp = [i.lower() for i in re.split(' ', str(key)) if i]
            temp.insert(0, '<s>')
            temp.append('</s>')
            for i in range(len(temp) - 1):
                token1 = temp[i]
                token2 = temp[i + 1]
                count = self.brand_dict[key] * self.brand_weight
                self.laplaceUnigramCounts[token1] += count
                self.laplaceBigramCounts[(token1, token2)] += count
                self.total += count
            self.total += count
            self.laplaceUnigramCounts[temp[-1]] += count

    def candidate_word(self, word):
        # """
        # Generate similar word for a given word
        # """
        suggests = [word]

        if word.isdigit():
            return suggests, len(suggests)
        elif self.is_spanish:
            suggests = self.spanish_dict.suggest(word)
        else:
            suggests = self.dict.suggest(word)
        suggests = [suggest.lower() for suggest in suggests][:3]
        suggests.append(word)
        suggests = list(set(suggests))
        #print(suggests)

        return suggests, len(suggests)

    def candidate_sentence(self, sentence):
        # """
        # Takes one sentence, and return all the possible sentences, and also return a dictionary of word : suggested number of words
        # """
        candidate_sentences = []
        words_count = {}
        for word in sentence:
            candidate_sentences.append(self.candidate_word(word)[0])
            words_count[word] = self.candidate_word(word)[1]

        candidate_sentences = list(itertools.product(*candidate_sentences))
        return candidate_sentences, words_count

    def correction_score(self, words_count, old_sentence, new_sentence):
        # """
        #   Take a old sentence and a new sentence, for each words in the new sentence, if it's same as the orginal sentence, assign 0.95 prob
        #   If it's not same as original sentence, give 0.05 / (count(similarword) - 1)
        # """
        score = 1
        for i in range(len(new_sentence)):
            if new_sentence[i] in words_count:
                score *= 0.95
            else:
                score *= (0.05 / (words_count[old_sentence[i]] - 1))
        return math.log(score)

    def score(self, sentence):
        # """
        #     Takes a list of strings as argument and returns the log-probability of the
        #     sentence using the stupid backoff language model.
        #     Use laplace smoothing to avoid new words with 0 probability
        # """
        score = 0.0
        if self.is_spanish == False:
            for i in range(len(sentence) - 1):
                if self.laplaceBigramCounts[(sentence[i], sentence[i + 1])] > 0:
                    score += math.log(self.laplaceBigramCounts[(sentence[i], sentence[i + 1])])
                    score -= math.log(self.laplaceUnigramCounts[sentence[i]])
                else:
                    score += (math.log(self.laplaceUnigramCounts[sentence[i + 1]] + 1) + math.log(0.4))
                    score -= math.log(self.total + len(self.laplaceUnigramCounts))
        else:
            for i in range(len(sentence) - 1):
                if self.laplaceSpanishBigramCounts[(sentence[i], sentence[i + 1])] > 0:
                    score += math.log(self.laplaceSpanishBigramCounts[(sentence[i], sentence[i + 1])])
                    score -= math.log(self.laplaceSpanishUnigramCounts[sentence[i]])
                else:
                    score += (math.log(self.laplaceSpanishUnigramCounts[sentence[i + 1]] + 1) + math.log(0.4))
                    score -= math.log(self.spanish_total + len(self.laplaceSpanishUnigramCounts))
        return score

    def return_best_sentence(self, old_sentence):
        # """
        #   Generate all candiate sentences and
        #   Calculate the prob of each one and return the one with highest probability
        #   Probability involves two part 1. correct probability and 2. language model prob
        #   correct prob : p(c | w)
        #   language model prob : use stupid backoff algorithm
        # """
        # gc.collect()
        bestScore = float('-inf')
        bestSentence = []

        # Code to detect if the sentence is in Spanish, if applicable
        languages = [Language.ENGLISH, Language.SPANISH]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
        if detector.detect_language_of(old_sentence) == Language.SPANISH:
            # print("detected as spanish: ", old_sentence)
            self.is_spanish = True
        else:
            self.is_spanish = False

        old_sentence = [word.lower() for word in old_sentence.split()]
        
        if old_sentence == [""]:
            return "", 0

        sentences, word_count = self.candidate_sentence(old_sentence)
        for new_sentence in sentences:
            new_sentence = list(new_sentence)
            score = self.correction_score(word_count, new_sentence, old_sentence)
            new_sentence.insert(0, '<s>')
            new_sentence.append('</s>')
            score += self.score(new_sentence)
            if score >= bestScore:
                bestScore = score
                bestSentence = new_sentence
        bestSentence = ' '.join(bestSentence[1:-1])
        return bestSentence, bestScore

# """
#   Can use this section of codes to test on individual inputs
# """
# time1=datetime.datetime.now()
# corrector = Sentence_Corrector('./big.txt')
# print(corrector.return_best_sentence(' urtains'))
# time2=datetime.datetime.now()
# print(time2 - time1)
