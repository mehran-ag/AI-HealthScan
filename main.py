import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("Data/Medical Data.csv")

cols = df.columns

my_labels = pd.unique(df["Label"])

df["Label"] = df["Label"].astype("category")
df["codes"] = df["Label"].cat.codes
codes = df["codes"].to_numpy()

n = len(df.columns)
myCols = df.columns[1:n - 2]

xx = df[myCols].to_numpy()
yy = df["Label"].tolist()

x1, x2, y1, y2 = train_test_split(xx, yy, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=4)

models = knn.fit(x1, y1)

ym = knn.predict(x2)

while True:

    p_name = input("Enter the patient's name: ")
    print("Enter a number for the severity of the symptom: 0-No symptom 1-mild, 2-moderate, 3-severe."
          "\nEnter these parameters carefully")
    n_features = len(myCols)
    x3 = np.zeros(n_features)
    for i in range(n_features):
        dd = input(f"Please enter the {myCols[i]}: ")
        x3[i] = float(dd)
    results = knn.predict(x3.reshape([1, n_features]))
    print(f"The current situation of {p_name} is  {results}")
    mm = input("Do you want to continue for other patient? y/n: ")
    if mm != "y":
        break