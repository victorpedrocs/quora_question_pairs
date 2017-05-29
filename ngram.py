from numpy import load, ones, save
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

#X = csr_matrix(load('./datasets/count-stemm-12g.npy').all())
#X = hstack((X, csr_matrix(ones((X.shape[0], 1)))))
X = load('./datasets/count-stemm-12g-300tsvd.npy')
y = load('./datasets/y.npy')

#pca = TruncatedSVD(n_components=50)
#X = pca.fit_transform(X)
#save('./datasets/count-stemm-12g-50tsvd', X)

out = open('./resultados/12g-pca300-1000-params-xb.txt', 'w')

#model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
#scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

#score = str("Log Loss (RF): %f (+/- %f)" % (scores.mean(), scores.std() * 2))
#out.write(score)

model = XGBClassifier(n_estimators=1000, max_depth=7, base_score=0.2, subsample=0.6)
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

score = str("Log Loss (XGB): %f (+/- %f)" % (scores.mean(), scores.std() * 2))
out.write(score)

out.close()