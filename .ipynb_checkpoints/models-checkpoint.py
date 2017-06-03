from numpy import load, ones, save
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from datetime import datetime
from glob import glob

def xgboost(X, y):
    model = XGBClassifier(n_estimators=1000, max_depth=7, base_score=0.2, subsample=0.6)
    return cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

def random_forest(X, y):
    model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    return cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

def bayes(X, y):
    model = naive_bayes.MultinomialNB()
    return cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')


def train_xgboost(X_train, y_train, _n_estimators):
    model = XGBClassifier(n_estimators=_n_estimators, max_depth=7, base_score=0.2, subsample=0.6)
    return model.fit(X_train, y_train)

def train_randomforest(X_train, y_train, _n_estimators):
    model = RandomForestClassifier(n_estimators=_n_estimators, n_jobs=-1, max_depth=3)
    return model.fit(X_train, y_train)

y = load('./npyarrays/y.npy')
out = open('./resultados/all_classifiers.txt', 'a')

for file in glob('./npyarrays/X/*.npy'):
    print('Training classifiers on', file)
    X = load(file)
    params = [100, 200, 500, 1000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    result = Parallel(n_jobs=8)(delayed(train_xgboost)(X_train, y_train, n_estimators) for n_estimators in params)

    for model in result:
        y_pred = model.predict(X_test)
        out.write(datetime.now().strftime('%H:%M:%S %Y-%m-%d\n'))
        out.write(str(model) + '\n\n')
        out.write(str(classification_report(y_test, y_pred)) + '\n\n')
        out.write("Acurracy: " + str(accuracy_score(y_test, y_pred)) + '\n\n')
        out.write("Log Loss: " + str(log_loss(y_test, y_pred)) + '\n\n')
        out.write(str(confusion_matrix(y_test, y_pred)) + '\n\n')

    result = Parallel(n_jobs=8)(delayed(train_xgboost)(X_train, y_train, n_estimators) for n_estimators in params)

    for model in result:
        y_pred = model.predict(X_test)
        out.write(datetime.now().strftime('%H:%M:%S %Y-%m-%d\n'))
        out.write(str(model) + '\n\n')
        out.write(str(classification_report(y_test, y_pred)) + '\n\n')
        out.write("Acurracy: " + str(accuracy_score(y_test, y_pred)) + '\n\n')
        out.write("Log Loss: " + str(log_loss(y_test, y_pred)) + '\n\n')
        out.write(str(confusion_matrix(y_test, y_pred)) + '\n\n')

        
out.close()