class trainData:
    # remove any user in the dataset that has less than 50 tweets
    def __init__(self, threshold=50):
        self.authors = dict()
        self.authorsList = []
        self.tweetList = []
        self.numOfAuthor = 0
        self.label = []
        self.data = []
        self.threshold = threshold
        self.__getAllAuthorsAndTweets__()


    def __getAllAuthorsAndTweets__(self):
        print("load in training data...")
        self.authors = dict()
        file = open("train_tweets.txt", 'r', encoding='utf-8').readlines()
        for line in file:
            tmp = line.split('\t')
            if self.authors.get(tmp[0]) is None:
                self.numOfAuthor += 1
                self.authors[tmp[0]] = []
                self.authorsList.append(tmp[0])

            # remove newline
            l = tmp[1].rstrip()
            #self.label.append(tmp[0])
            #self.data.append(tmp[1])
            self.authors[tmp[0]].append(l)
            #self.tweetList.append(l)

        self.generateLabelAndData()

        print("load finish!")

    def generateLabelAndData(self):

        self.label = []
        self.data = []

        for k in self.authors.keys():
            if self.authors[k].__len__() < self.threshold:
                self.authors[k] = []

            if self.authors[k].__len__() > 0:
                for s in self.authors[k]:
                    self.label.append(k)
                    self.data.append(s)
                    self.tweetList.append(s)

    def getAllTweetFromAuthor(self, userID):
        return self.authors[userID]

    def getAllTweetInList(self):
        return self.tweetList

    def getLabelsAndrawData(self):
        return self.label, self.data

    def unloadData(self):
        self.authors = None
        self.authorsList = None
        self.tweetList = None
        self.numOfAuthor = None
        self.label = None
        self.data = None

    def getTweetNumOfAuthor(self, userID):
        return self.authors[userID].__len__()

class testData:

    def __init__(self):
        self.tweetList = []
        self.__loadTweets__()

    def __loadTweets__(self):
        file = open("test_tweets_unlabeled.txt", 'r', encoding='utf-8').readlines()

        for line in file:
            # remove newline
            l = line.rstrip()
            self.tweetList.append(l)

    def getAllTweets(self):
        return self.tweetList

    def getTweetsWithNum(self, num):
        if num > 0:
            return self.tweetList[:num]
        return []
