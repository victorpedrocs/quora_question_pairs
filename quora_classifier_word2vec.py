from pandas import read_csv
from numpy import array, zeros, float32, divide, add
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import RegexpTokenizer

# glove2word2vec('datasets/glove.6B.50d.txt', 'datasets/glove.6B.50d.word2vec')

dataset = open('datasets/dataset_concat.csv', 'r')
dataset = read_csv(dataset, header=None, delimiter='\t')

print(dataset.shape)

x = zeros((dataset.shape[0], 50), dtype='float32')
y =  array(dataset.iloc[:,1]).ravel()

print(x.shape)

model = KeyedVectors.load_word2vec_format('datasets/glove.6B.50d.word2vec', binary=False)

tokenizer = RegexpTokenizer(r'\w+')

row_index = 0

for row in dataset.itertuples():

    words = tokenizer.tokenize(row[1])

    filtered_words = [word for word in words if word in model.vocab]

    words_count = float32(len(filtered_words))

    for word in filtered_words:

        word_vector = model.word_vec(word)
        x[row_index] = add(x[row_index], word_vector)

    # x[row_index] = divide(x[row_index], words_count)

    row_index += 1

print(x.shape)
print(y.shape)

# classifier = naive_bayes.MultinomialNB()

classifier = RandomForestClassifier(n_estimators=15, n_jobs=-1)

scores = cross_val_score(classifier, x, y, cv=5, n_jobs=-1, scoring='log_loss')

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))