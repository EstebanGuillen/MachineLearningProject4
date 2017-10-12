import pandas as pd
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.cross_validation import StratifiedKFold


from sklearn.linear_model import SGDClassifier


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from sklearn import preprocessing
from scipy import stats

from sklearn.grid_search import GridSearchCV

pd.set_option('mode.use_inf_as_null', True)

labels = ["0","1","2","3","4","5","6","7","8","9"]


def train(X,y,pipe,classifier_features):
    
    print ('')
    print ('')
    print('***** '+ classifier_features + ' *****')
    print('')

    skf = StratifiedKFold(y_train, n_folds=10, random_state=1)

    scores = []
    confusion_matrix = np.zeros((10,10),dtype=np.uint8)
    for train_index, test_index in skf:
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        pipe.fit(X_train_fold, y_train_fold)
        y_predict = pipe.predict(X_test_fold)
        scores.append(pipe.score(X_test_fold, y_test_fold))
    
        for p,r in zip(y_predict, y_test_fold):
            confusion_matrix[p,r] = confusion_matrix[p,r] + 1
        
    
    mean, sigma = np.mean(scores), np.std(scores)
    
    conf_int = stats.norm.interval(0.95, loc=mean, scale=sigma / np.sqrt(len(scores)))
    print('Average CV accuracy: %.6f +/- %.6f' % (np.mean(scores),np.mean(scores)-conf_int[0]))
    
    print('CV 95 percent confidence interval:', conf_int)
    print(confusion_matrix)

    
    
def classify(X,y,X_test,y_test,pipe,classifier_features):
    #train(X,y,pipe, classifier_features)
    pipe.fit(X, y)
    print('Test Accuracy %s: %.6f' % (classifier_features, pipe.score(X_test, y_test)) )
    
def validate(X,y,X_validate, pipe, file_name):
    
    with open(file_name, "w") as f:
        f.write('\"ImageId\",\"Label\"\n')
        pipe.fit(X,y)
        for i in range(0,len(X_validate)):
            sample = X_validate[i,:].reshape(1,-1)
            prediction = pipe.predict(sample)
            line = str(i+1) + "," + '\"' + str(labels[int(prediction[0])]) + '\"' + "\n"
            f.write(line)


df = pd.read_csv('train.csv', skiprows=1, header=None)
#df = df.dropna(axis=0)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

#X = preprocessing.scale(X)


print("splitting train data")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=1)


df_validate = pd.read_csv('test.csv', skiprows=1, header=None)
#df_validate = df_validate.dropna(axis=0)

X_validate = df_validate.iloc[:, :].values

print ("Training SVM - PCA=50 C=5")      
pipe = Pipeline([('pca',PCA(n_components=50)), ('scl', StandardScaler()), ('clf',SVC(C=5.0))])
train(X,y,pipe,"Training SVM - PCA=50 C=5")
classify(X_train,y_train,X_test,y_test,pipe,"Training SVM - PCA=50 - C=5")
#validate(X,y,X_validate,pipe,'predictions-svm-pca50-C20.txt')

