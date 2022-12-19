#linear regression predict values
from keras.models import  Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

x,y = make_regression(n_samples=100, n_features=2, random_state=1, noise=0.1)
print(x)
print(y)
scalarx, scalry = MinMaxScaler(), MinMaxScaler()
scalarx.fit(x)
scalry.fit(y.reshape(100, 1))
x = scalarx.transform(x)
y = scalry.transform(y.reshape(100, 1))

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, verbose=0)
xnew, a = make_regression(n_samples=3, n_features=2, random_state=1, noise=0.1)
xnew = scalarx.transform(xnew)
ynew = model.predict(xnew)
for i in range(len(xnew)):
    print("x %s predecited %s"%(xnew[i], ynew[i]))