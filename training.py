# TRAINING FILE
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

data_dict = pickle.load(open('./data/processed/data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

history = model.fit(x_train, y_train)

y_predict = model.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)

print('{}% of samples were classified correctly! '.format(accuracy * 100))

f = open('./data/models/model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# graficas
if not os.path.exists('graficas'):
    os.makedirs('graficas')


confusion_mtx = confusion_matrix(y_test, y_predict)

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
           'w', 'x', 'y', 'z']

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("prediccion")
plt.ylabel("real")
plt.savefig("graficas/confusion_matrix.png")
plt.show()
