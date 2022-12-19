#mutliclass classification predict values
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=1)
scalar = MinMaxScaler()
model = Sequential()
scalar.fit(x)
x = scalar.transform(x)
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x, y, epochs=500)
xnew, yreal = make_blobs(n_samples=3, n_features=2, centers=2, random_state=1)
xnew = scalar.transform(xnew)
ynew = model.predict_step(xnew)
for i in range (len(xnew)):
    print("X = %s predicted = %s desired = %s"%(xnew[i], ynew[i], yreal[i]))