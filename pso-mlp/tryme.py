### Adjusting the weights for each connection between the hidden layers of neurons of MLP using PSO
#import necessary libraries
import numpy as np
from PSOMLP import PSOMLP
import numpy as np
from sklearn.metrics import precision_score,accuracy_score,classification_report,confusion_matrix


# split dataset
from sklearn.datasets import load_breast_cancer
DATA = load_breast_cancer()
X = DATA.data
Y = DATA.target

NUM_SAMPLES = 500
NUM_TRAINSET = 200
indices = list(range(NUM_SAMPLES))
#print(len(indices))
np.random.shuffle(indices)
train_indices = indices[:NUM_TRAINSET]
test_indices = indices[NUM_TRAINSET:]
#Split test set and train set
X_train = X[train_indices, :]
X_test = X[test_indices, :]
Y_train = Y[train_indices]
Y_test = Y[test_indices]
print(Y_test)
pso = PSOMLP(hlayers=(10, ))
mlp = pso.fit(X_train, Y_train, iterations=100)
mlp_predicted = mlp.predict(X_test)

print(mlp_predicted)

print("Accuracy for trainning data:", 100 * mlp.score(X_train, Y_train))

#importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(Y_test, mlp_predicted)
print('Confusion Matrix\n')
print(confusion)

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(Y_test, mlp_predicted)))

print('Micro Precision: {:.2f}'.format(precision_score(Y_test, mlp_predicted, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(Y_test, mlp_predicted, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(Y_test, mlp_predicted, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(Y_test, mlp_predicted, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(Y_test, mlp_predicted, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(Y_test, mlp_predicted, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(Y_test, mlp_predicted, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(Y_test, mlp_predicted, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(Y_test, mlp_predicted, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(Y_test, mlp_predicted, target_names=['Class 0', 'Class 1']))

