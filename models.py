from numpy import load, ones, save
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split

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

X_w2v = load('./datasets/300d-1-x.npy')
y_w2v = load('./datasets/y.npy')
out_word2vec = open('./resultados/word2vec.txt', 'w')

X_ng = load('./datasets/count-stemm-12g-300tsvd.npy')
y_ng = load('./datasets/y.npy')
out_ngram = open('./resultados/ngram.txt', 'w')


params = [100, 200, 500, 1000]

X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(X_w2v, y_w2v, test_size=0.2, random_state=25)
X_train_ng, X_test_ng, y_train_ng, y_test_ng = train_test_split(X_ng, y_ng, test_size=0.2, random_state=25)

result = Parallel(n_jobs=8)(delayed(train_xgboost)(X_train_w2v, y_train_w2v, n_estimators) for n_estimators in params)

for model in result:
    y_pred = model.predict(X_test_w2v)
    out_word2vec.write(str(model) + '\n\n')
    out_word2vec.write(str(classification_report(y_test_w2v, y_pred)) + '\n\n')
    out_word2vec.write("Acurracy: " + str(accuracy_score(y_test_w2v, y_pred)) + '\n\n')
    out_word2vec.write("Log Loss: " + str(log_loss(y_test_w2v, y_pred)) + '\n\n')
    out_word2vec.write(str(confusion_matrix(y_test_w2v, y_pred)) + '\n\n')

result = Parallel(n_jobs=8)(delayed(train_xgboost)(X_train_ng, y_train_ng, n_estimators) for n_estimators in params)

for model in result:
    y_pred = model.predict(X_test_ng)
    out_ngram.write(str(model) + '\n\n')
    out_ngram.write(str(classification_report(y_test_ng, y_pred)) + '\n\n')
    out_ngram.write("Acurracy: " + str(accuracy_score(y_test_ng, y_pred)) + '\n\n')
    out_ngram.write("Log Loss: " + str(log_loss(y_test_ng, y_pred)) + '\n\n')
    out_ngram.write(str(confusion_matrix(y_test_ng, y_pred)) + '\n\n')


result = Parallel(n_jobs=8)(delayed(train_randomforest)(X_train_ng, y_train_ng, n_estimators) for n_estimators in params)

for model in result:
    y_pred = model.predict(X_test_ng)
    out_ngram.write(str(model) + '\n\n')
    out_ngram.write(str(classification_report(y_test_ng, y_pred)) + '\n\n')
    out_ngram.write("Acurracy: " + str(accuracy_score(y_test_ng, y_pred)) + '\n\n')
    out_ngram.write("Log Loss: " + str(log_loss(y_test_ng, y_pred)) + '\n\n')
    out_ngram.write(str(confusion_matrix(y_test_ng, y_pred)) + '\n\n')

result = Parallel(n_jobs=8)(delayed(train_randomforest)(X_train_w2v, X_train_w2v, n_estimators) for n_estimators in params)

for model in result:
    y_pred = model.predict(X_test_w2v)
    out_word2vec.write(str(model) + '\n\n')
    out_word2vec.write(str(classification_report(y_test_w2v, y_pred)) + '\n\n')
    out_word2vec.write("Acurracy: " + str(accuracy_score(y_test_w2v, y_pred)) + '\n\n')
    out_word2vec.write("Log Loss: " + str(log_loss(y_test_w2v, y_pred)) + '\n\n')
    out_word2vec.write(str(confusion_matrix(y_test_w2v, y_pred)) + '\n\n')
