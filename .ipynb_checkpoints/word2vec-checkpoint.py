from numpy import load, ones, divide
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm
from sklearn.model_selection import cross_val_score

X = load('./datasets/300d-1-x.npy')
#X = hstack((X, csr_matrix(ones((X.shape[0], 1)))))
y = load('./datasets/y.npy')

out = open('./resultados/test_norm_100.txt', 'w')

#model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
#scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

#score = str("Log Loss (RF): %f (+/- %f)" % (scores.mean(), scores.std() * 2))
#out.write(score)

for row in range(X.shape[0]):
    X[row] = divide(X[row], norm(X[row]))

model = XGBClassifier(n_estimators=100)
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

#model = LogisticRegression(solver='sag', n_jobs=-1, max_iter=100000)
#scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

score = str("Log Loss (XGB): %f (+/- %f)" % (scores.mean(), scores.std() * 2))
out.write(score)

out.close()