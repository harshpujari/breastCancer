import os; import sys; os.system(f"{sys.executable} -m pip install -U --quiet deepchecks")

import pandas as pd
import sklearn as sk
from deepchecks.tabular.checks import TrainTestFeatureDrift
from deepchecks.tabular import Dataset
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns

pip install matplotlib==3.1.1

path_to_train_data = "train.csv"
path_to_test_data = "test.csv"
train_dataset = Dataset(pd.read_csv(path_to_train_data), label="target", cat_features=[], label_type="classification_label")
test_dataset = Dataset(pd.read_csv(path_to_test_data), label="target", cat_features=[], label_type="classification_label")

check = TrainTestFeatureDrift(columns=["mean radius"], show_categories_by="largest_difference")
result = check.run(train_dataset, test_dataset)
result.show()

trainData = pd.read_csv('train.csv')
xTrain = trainData.iloc[:,:-1].values
yTrain = trainData.iloc[:, -1].values

testData = pd.read_csv('test.csv')
xTest = testData.iloc[:,:-1].values
yTest = testData.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(xTrain, yTrain)

yPredProb = regressor.predict(xTest)

print(yPredProb)

yPredTF = np.round(yPredProb)

print(yPredTF)

print(np.concatenate((yPredTF.reshape(len(yPredTF),1), yTest.reshape(len(yTest),1)),1))

from sklearn import metrics
confusionMatrix = metrics.confusion_matrix(yTest,yPredTF)

from sklearn.metrics import zero_one_loss
zol = zero_one_loss
zoll = zol(yTest, yPredTF, normalize=True, sample_weight=None)
print("Percentage of wrong prediction: ",zoll*100,"%")

print("Accuracy of the model is :",100-zoll*100)

print(confusionMatrix)

labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)

akws = {"ha": 'center',"va": 'center'}
ax = sns.heatmap(confusionMatrix, annot= True, fmt="d", annot_kws=akws, linewidths=.5, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()
