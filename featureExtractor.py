import re
import matplotlib.pyplot as plt
from fileReader import trainData
import numpy as np
from nltk import TweetTokenizer
import emot
from w2v_processing import w2vAndGramsConverter


class extractor:
    def __init__(self):
        pass

    def isRetweet(self, line):
        if re.search(r"[R|r][T|t]:?\s@handle:?\s?", line):
            return 1.
        return 0.

    def mentionWordRatio(self, line):
        l = re.findall(r"(?<![R|r][T|t]\s)@handle", line)
        w = self.totalWord(line)
        return l.__len__() / w

    def totalWord(self, line):
        return line.split(" ").__len__()

    def numOfURL(self, line):
        l = re.findall("http", line)
        return l.__len__()

    def hashtagWordRatio(self, line):
        l = re.findall(r"(?<!&)#", line)
        w = self.totalWord(line)
        return l.__len__() / w

    def numOfMoney(self, line):
        l = re.findall(r"\$[1-9][0-9]*", line)
        return l.__len__()

    def lineToVector(self, line):
        return [
            self.isRetweet(line),
            self.mentionWordRatio(line),
            self.totalWord(line),
            self.hashtagWordRatio(line),
            self.numOfMoney(line)
        ]

    def str(self, vec):
        s = ""
        for x in vec:
            s += str(x) + ','
        s = s[:-1]
        return s

    def batchToVector(self, d, usr_flag, save=False, file_name="data"):
        print("convert raw data to vector.....")
        if save:
            print("saving enabled, writing to file: " + file_name)
            save_file = open(file_name, 'w', encoding='utf-8')
            if usr_flag:
                for k in d.keys():
                    for l in d[k]:
                        save_file.write(k + '\t' + self.str(self.lineToVector(l)) + '\n')
            else:
                for l in d:
                    save_file.write(self.str(self.lineToVector(l)) + '\n')

            save_file.close()
            print("saving complete!")

        else:
            if usr_flag:
                new_d = d
                for k in d.keys():
                    for l in d[k]:
                        new_d[k] = self.lineToVector(l)

                return new_d

            else:
                d_arr = []
                for l in d:
                    d_arr.append(self.lineToVector(l))

                return d_arr

    """
    ===============================
            Binary/n Features
    +++++++++++++++++++++++++++++++
    """

    def hasEmoji(self, line):
        result = emot.emoticons(line)
        if result.__len__() > 0:
            return 1
        return 0

    # 1: less than 10 tokens(include)
    # 2: less than 20 tokens(include)
    # 3: less than 30 tokens(include)
    # 4: less than 40 tokens(include)
    # 5: more than 40 tokens(exclude)
    def tweetLength(self, line):
        # normalize the line
        w2vLib = w2vAndGramsConverter()
        line = w2vLib.normalizeSentence(line)

        # tokenize sentence
        tnz = TweetTokenizer()
        tokens = tnz.tokenize(line)

        if tokens.__len__() <= 10:
            return 1
        elif tokens.__len__() <= 20:
            return 2
        elif tokens.__len__() <= 30:
            return 3
        elif tokens.__len__() <= 40:
            return 4
        else:
            return 5

    # 0: no URL
    # 1: 1 URL
    # 2: 2 URL
    # 3: 3 URL
    # 4: more than 3 URL
    def getURLFeature(self, line):
        return self.numOfURL(line)

    def isRT(self, line):
        return self.isRetweet(line)

    def containMoney(self, line):
        if self.numOfMoney(line) > 0:
            return 1
        return 0

    def hasCaptialWord(self, line):
        return re.findall(r"\s[A-Z]{2,}", line).__len__()
        """
        if re.search(r"\s[A-Z]{2,}", line):
            return 1
        return 0
        """

    def hasNowPlaying(self, line):
        if re.search(r"Now playing:\s", line):
            return 1
        return 0

    def useOfPuncs(self, line):
        if re.search(r".{2,}", line) and re.search(r"!{2,}", line):
            return 3
        elif re.search(r".{2,}", line):
            return 2
        elif re.search(r"!{2,}", line):
            return 1
        return 0

    def hasNum(self, line):

        if re.search(r"\s[0-9]*\s", line):
            return 1
        return 0

    def hasMention(self, line):
        return re.findall(r"(?<![R|r][T|t]\s)@handle", line).__len__()
        """
        if re.search(r"(?<![R|r][T|t]\s)@handle", line):
            return 1
        return 0
        """

    def hasRepeatLetters(self, line):
        if re.search(r"\s[a-zA-Z]*([a-zA-Z])\1{1}[a-zA-Z]*\s", line):
            return 1
        return 0

    def hasHash(self, line):
        return re.findall(r"#\w+", line).__len__()
        """
        if re.search(r"#\w+", line):
            return 1
        return 0
        """

    def lineToFixFeatureVec(self, line):
        return [
            self.tweetLength(line),
            self.isRT(line),
            self.getURLFeature(line),
            self.hasMention(line),
            self.hasRepeatLetters(line),
            self.hasNum(line),
            self.containMoney(line),
            self.useOfPuncs(line),
            self.hasCaptialWord(line),
            self.hasEmoji(line),
            self.hasHash(line),
        ]

    def batchProduceFixFeatureVec(self, lines):
        print("transfering rawdata into vectors.....")
        retArr = []
        for l in lines:
            retArr.append(self.lineToFixFeatureVec(l))

        print("transfer complete!")
        return retArr

    # 0: no capital of word
    # 1: 50% of words are start with Capital
    def COWvalue(self, line):
        l = re.findall(r'[A-Z][a-z]+', line)
        w = re.findall(r'[a-zA-Z]+\s', line)
        v = l.__len__() / w.__len__()
        if v > 0.5:
            return 1
        else:
            return 0
