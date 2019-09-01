from fileReader import trainData, testData
from featureExtractor import extractor
from sklearn import svm
import pickle

# create trainingData and feature extractor
train = trainData(threshold=10)
e = extractor()

# read training data
label, raw = train.getLabelsAndrawData()

data = e.batchToVector(raw, usr_flag=False)

# clear training data for memory saving
train.unloadData()

# create svm model
print("init model....")
clf = svm.SVC(gamma='scale', verbose=False)

print("training model.....")
# train model
clf.fit(data, label)

print("finished training!!!!")
# save model

print("saving model...")
with open("svmModel.pkl", 'wb') as f:
    pickle.dump(clf, f)
