#binary classification
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
dataset = loadtxt('diabetes.csv', delimiter=",")
x = dataset[:, 0:8]
y = dataset[:,8]
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=10)
predict = model.predict_step(x)
print(predict)
_, accuracy = model.evaluate(x, y)
print("Accuracy:", (accuracy*100))