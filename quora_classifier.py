#%%
from pandas import read_csv
from numpy  import array
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

dataset = open('datasets/dataset_simple.csv', 'r')
#%%
dataset = read_csv(dataset, header=None, delimiter='\t')

print(dataset.shape)
#%%
x = dataset.iloc[:,0]
y = array(dataset.iloc[:,1]).ravel()

#ngram_range=(1,3)

vectorizer = CountVectorizer(min_df=1)
x = vectorizer.fit_transform(x)

print(x.shape)
print(y.shape)
#%%
# classifier = naive_bayes.MultinomialNB()

classifier = RandomForestClassifier(n_estimators=15, n_jobs=-1)

scores = cross_val_score(classifier, x, y, cv=5, n_jobs=-1, scoring='neg_log_loss')

print("Log Loss: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
#%%
