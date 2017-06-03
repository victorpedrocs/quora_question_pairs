from numpy import load, ones, save
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

#X = csr_matrix(load('./datasets/count-stemm-12g.npy').all())
#X = hstack((X, csr_matrix(ones((X.shape[0], 1)))))
X = load('./datasets/count-stemm-12g-300tsvd.npy')
y = load('./datasets/y.npy')

#pca = TruncatedSVD(n_components=50)
#X = pca.fit_transform(X)
#save('./datasets/count-stemm-12g-50tsvd', X)

out = open('./resultados/xgboost-12g-300svd-1000.txt', 'w')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

def train_model(X_train, y_train, _n_estimators):
    model = XGBClassifier(n_estimators=_n_estimators, max_depth=7, base_score=0.2, subsample=0.6)
    #model = RandomForestClassifier(n_estimators=_n_estimators, n_jobs=-1)
    return model.fit(X_train, y_train)

#params = [100, 200, 500, 1000]

#result = Parallel(n_jobs=8)(delayed(train_model)(X_train, y_train, n_estimators) for n_estimators in params)

#for model in result:
#    y_pred = model.predict(X_test)
#    out.write(str(model) + '\n\n')
#    out.write(str(classification_report(y_test, y_pred)) + '\n\n')
#    out.write("Accuracy: " + str(accuracy_score(y_test, y_pred)) + '\n\n')
#    out.write("Log Loss: " + str(log_loss(y_test, y_pred)) + '\n\n')
#    out.write(str(confusion_matrix(y_test, y_pred)) + '\n\n')

#model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
#scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

#score = str("Log Loss (RF): %f (+/- %f)" % (scores.mean(), scores.std() * 2))
#out.write(score)

model = XGBClassifier(n_estimators=1000, max_depth=7, base_score=0.2, subsample=0.6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
out.write(str(model) + '\n\n')
out.write(str(classification_report(y_test, y_pred)) + '\n\n')
out.write("Accuracy: " + str(accuracy_score(y_test, y_pred)) + '\n\n')
out.write("Log Loss: " + str(log_loss(y_test, y_pred)) + '\n\n')
out.write(str(confusion_matrix(y_test, y_pred)) + '\n\n')

#model = XGBClassifier(n_estimators=1000, max_depth=7, base_score=0.2, subsample=0.6)
#scores = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='neg_log_loss')
#score = str("Log Loss (XGB): %f (+/- %f)" % (scores.mean(), scores.std() * 2))
#out.write(score)

out.close()