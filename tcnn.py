import tensorflow as tf
from tensorflow import keras
from gensim.models import Word2Vec
from w2v_processing import w2vAndGramsConverter
from fileReader import trainData

model = keras.Sequential([
    keras.layers.Dense(300, input_shape=(0,1)),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(10000, activation='softmax')
])

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# get training data
clf = Word2Vec.load("w2v.model")

c = w2vAndGramsConverter()
train = trainData(threshold=0)
label, data = train.getLabelsAndrawData()
train.unloadData()
#c.removeHighAndLowFrequencyWords(data)
#c.trainW2V()


while not c.batchFlag:
    label, data = c.convertDataToVec(data, label)
    model.fit(data, label, epochs=5)

model.save("nn.model")