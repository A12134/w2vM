from fileReader import testData
from featureExtractor import extractor
import pickle

def predictUsingSVMModel(lines):
    # create output file
    output = open("output.csv", 'w', encoding='utf-8')
    output.write("Id,Predicted\n")

    # create extractor
    ext = extractor()

    # vectorize test data
    list = ext.batchToVector(lines, usr_flag=False)

    # load in SVM model
    with open("svmModel.pkl", 'rb') as file:
        model = pickle.load(file)

    id = 0
    for l in list:
        id += 1
        ans = model.predict([l])
        output.write(str(id)+','+ans[0]+'\n')
        print("progress: " + str(id*100/list.__len__()) + "%")

# load testdata
t = testData()
data = t.getAllTweets()

predictUsingSVMModel(data)