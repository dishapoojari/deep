#predciting class
from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np

x, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
model = Sequential()
scalar.fit(x)
x = scalar.transform(x)
model.add(Dense(4, activation='relu', input_dim=2))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x, y, epochs=500)
xnew, yreal = make_blobs(n_samples=3, n_features=2, centers=2, random_state=1)
xnew = scalar.transform(xnew)
yclass = model.predict_step(xnew)
predict_prob = model.predict([xnew])
predict_class = np.argmax(predict_prob, axis=1)
for i in range(len(xnew)):
    print("x- %s, predicted probability- %s, predicted class - %s"%(xnew[i], predict_class[i], yclass[i]))