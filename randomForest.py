import pandas as pd
import numpy as np
from fileReader import trainData
from sklearn.model_selection import train_test_split
from featureExtractor import extractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

td = trainData(threshold=50)
label,raw = td.getLabelsAndrawData()

#process data
ext = extractor()
data = ext.batchProduceFixFeatureVec(raw)
td.unloadData()

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

#gnb = GaussianNB()
#gnb.fit(X_train, y_train)
#y_pred = gnb.predict(X_test)



# train model
regressor = RandomForestClassifier(n_estimators=50, criterion='entropy', min_samples_split=40, verbose=2)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# print output
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
