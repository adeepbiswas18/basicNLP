import Doc2Vec, Utilities
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection

#training and tesing of the SVM classifier
def trainTestSVM(trainData, trainLabels, testData, testLabels):
    clf = svm.SVC(decision_function_shape='ovr', C=100, gamma=0.9, kernel='poly')
    clf.fit(trainData, trainLabels)
    return clf.score(testData, testLabels)

#training and testing of the MLP classifier
def trainTestNN(trainData, trainLabels, testData, testLabels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30), random_state=1)
    clf.fit(trainData, trainLabels)
    return clf.score(testData, testLabels)

#generating and sending the features and labels to the classifier models
def trainTest(train, test, trainFn=trainTestSVM):
    vectorizer = Utilities.getVectorizer()
    X_train = vectorizer.fit_transform(train['Contents'])
    X_test = vectorizer.transform(test['Contents'])
    return trainFn(X_train, train['Labels'], X_test, test['Labels'])

train = Doc2Vec.getTrainTokens()
test = Doc2Vec.getTestTokens()

print("ACCURACY SCORE: ",trainTest(train, test, trainTestSVM))
