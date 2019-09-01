from gensim.models import Word2Vec
from nltk import ngrams
from nltk import TweetTokenizer
from collections import OrderedDict
from fileReader import trainData
import operator
import re
import math
import numpy as np

class w2vAndGramsConverter:

    def __init__(self):
        self.model = Word2Vec(size=300, workers=5)
        self.two_gram_list = []
        self.three_gram_list = []
        self.five_gram_list = []
        self.dictionary = dict()
        self.lines = []
        self.indexTracking = 0
        self.batchFlag = False

    def normalizeSentence(self, line):
        # remove newline
        line = line.rstrip()
        # remove Retweet
        line = re.sub(r"[R|r][T|t]:?\s@handle:?\s", "", line)
        # remove mentions
        line = re.sub(r"\s@[0-9a-zA-Z]+\s", "", line)
        # remove punctuation
        line = re.sub(r"\.\s", " ", line)

        return line

    def removeHighAndLowFrequencyWords(self, lines, percentage=0.4):
        tk = TweetTokenizer()
        dictionary = OrderedDict()

        # create dictionary
        for line in lines:
            l = tk.tokenize(self.normalizeSentence(line))
            self.lines.append(l)
            for token in l:
                if len(token) > 1 or re.search('\w', token):
                    if dictionary.get(token) is None:
                        dictionary[token] = 1
                    else:
                        dictionary[token] += 1

        # remove high frequency and low frequency words
        dictionary = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=False)

        while dictionary[0][1] < 5:
            del dictionary[0]

        index = math.floor(dictionary.__len__()*percentage)
        for i in range(index):
            del dictionary[0]
            del dictionary[-1]
        self.dictionary = dictionary

    def trainW2V(self):
        self.model.build_vocab(self.lines, progress_per=5000)
        self.model.train(self.lines, total_examples=self.model.corpus_count, epochs=200, report_delay=1)
        self.model.save("w2v.model")


    """
    =========================================
    NGrams methods
    =========================================
    """

    def generateTwoGram(self, line):
        return ngrams(line, 2)

    def generateThreeGram(self, line):
        return ngrams(line, 3)

    def generateFiveGram(self, line):
        return ngrams(line, 5)

    def createTwoGramVocab(self, lines):
        twoGram = dict()

        # process all sentences into 2 grams
        for l in lines:
            tmp = self.generateTwoGram(l)

            for token in tmp:
                if twoGram.get(token) is None:
                    twoGram[token] = 1
                else:
                    twoGram[token] += 1

        self.two_gram_list = [x for x in self.__removeHighAndLow(twoGram).items()]

    def createThreeGramVocab(self, lines):
        threeGram = dict()

        # process all sentences into 3 grams
        for l in lines:
            tmp = self.generateTwoGram(l)

            for token in tmp:
                if threeGram.get(token) is None:
                    threeGram[token] = 1
                else:
                    threeGram[token] += 1

        self.three_gram_list = [x for x in self.__removeHighAndLow(threeGram).items()]

    def createFiveGramVocab(self, lines):
        FiveGram = dict()

        # process all sentences into 3 grams
        for l in lines:
            tmp = self.generateTwoGram(l)

            for token in tmp:
                if FiveGram.get(token) is None:
                    FiveGram[token] = 1
                else:
                    FiveGram[token] += 1

        self.Five_gram_list = [x for x in self.__removeHighAndLow(FiveGram).items()]

    def __removeHighAndLow(self, gr, percentage=0.2):
        od = OrderedDict(sorted(gr.items(), key=operator.itemgetter(1), reverse=True))
        index = math.floor(od.__len__()*0.2)
        for i in range(index):
            del od[0]
            del od[-1]

        return od

    def convertDataToVec(self, data, labels, batchSize=5000):
        if data.__len__() - self.indexTracking < batchSize:
            batchSize = data.__len__() - self.indexTracking
            self.batchFlag = True

        clf = Word2Vec.load("w2v.model")
        d = np.array([])
        counts = 0
        for line in data[self.indexTracking:]:
            if counts == batchSize:
                break
            counts += 1
            tmp = np.array([0]*300)
            tk = TweetTokenizer()
            l = tk.tokenize(self.normalizeSentence(line))
            count = 0
            for w in l:
                count += 1
                try:
                    s = clf.wv.get_vector(w)
                    s = np.array(s)
                    tmp = np.add(tmp, s)
                except:
                    continue

            tmp = tmp/count
            d = np.concatenate((d, tmp))

        l = self.convertLabelToVec(labels, batchSize)
        self.indexTracking += batchSize

        return l, d

    def convertLabelToVec(self, labels, batchSize=5000):
        d = np.array([])
        count = 0
        for s in labels[self.indexTracking:]:
            if count == batchSize:
                break
            count += 1

            tmp = [0]*10000
            try:
                tmp[int(s)] = 1
            except:
                print("break")
            d = np.concatenate((d, tmp))
        return d


