#XOR Problem
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
x = np.array([[0.,0.,],[0.,1.,],[1.,0.,],[1.,1.,]])
y = np.array([0.,1.,1.,0,])

model = Sequential()
model.add(Dense(units=2, activation='relu', input_dim=2))
model.add(Dense(activation='sigmoid', units=1))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=4)
print("final weight")
print(model.get_weights())
print(model.predict(x, batch_size=4))