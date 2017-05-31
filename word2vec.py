from numpy import load, ones, divide
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

X = load('./datasets/300d-1-x.npy')
#X = hstack((X, csr_matrix(ones((X.shape[0], 1)))))
y = load('./datasets/y.npy')

out = open('./resultados/xgboost-word2vec.txt', 'w')

for row in range(X.shape[0]):
    X[row] = divide(X[row], norm(X[row]))
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

def train_model(X_train, y_train, _n_estimators):
    model = XGBClassifier(n_estimators=_n_estimators, max_depth=7, base_score=0.2, subsample=0.6)
    return model.fit(X_train, y_train)

params = [100, 200, 500, 1000]

result = Parallel(n_jobs=8)(delayed(train_model)(X_train, y_train, n_estimators) for n_estimators in params)

for model in result:
    y_pred = model.predict(X_test)
    out.write(str(model) + '\n\n')
    out.write(str(classification_report(y_test, y_pred)) + '\n\n')
    out.write("Acurracy: " + str(accuracy_score(y_test, y_pred)) + '\n\n')
    out.write("Log Loss: " + str(log_loss(y_test, y_pred)) + '\n\n')
    out.write(str(confusion_matrix(y_test, y_pred)) + '\n\n')

#model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
#scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

#score = str("Log Loss (RF): %f (+/- %f)" % (scores.mean(), scores.std() * 2))
#out.write(score)

#model = XGBClassifier(n_estimators=100)
#scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

#score = str("Log Loss (XGB): %f (+/- %f)" % (scores.mean(), scores.std() * 2))
#out.write(score)

out.close()