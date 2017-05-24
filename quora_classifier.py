from pandas import read_csv
from numpy  import array
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

dataset = open('dataset_concat_stemm.csv', 'r')
dataset = read_csv(dataset, header=None, delimiter='\t')

print(dataset.shape)

x = dataset.iloc[:,0]
y = array(dataset.iloc[:,1]).ravel()

tfidf_vectorizer = TfidfVectorizer(min_df=1)
x = tfidf_vectorizer.fit_transform(x)

print(x.shape)
print(y.shape)

# classifier = naive_bayes.MultinomialNB()

classifier = RandomForestClassifier(n_estimators=5, n_jobs=-1)

scores = cross_val_score(classifier, x, y, cv=2, n_jobs=-1)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



