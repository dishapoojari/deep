#Regularization
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
data = pd.read_csv("Iris.csv", names= names)
print(data.head(10))
x = data.drop("Species")
#x.reshape(-1, 1)
y = data["Species"]
x_train , x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(f"Before pca Train:{x_train} tEST: {x_test}")