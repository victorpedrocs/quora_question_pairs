from pandas import read_csv
from numpy import array, zeros, float32, add  # , divide
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn import naive_bayes
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# from sklearn import neighbors

from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import SVC
import time

start = time.time()
print("Open Dataset...")
dataset = open('datasets/dataset_simple_0.1.csv', 'r')
dataset = read_csv(dataset, header=None, delimiter='\t')

print("DataFrame(lines, cols) ", dataset.shape)

print("Splitting feature and label...")
x = zeros((dataset.shape[0], 300), dtype='float32')
y = array(dataset.iloc[:, 1]).ravel()

print("Features shape: ", x.shape)

print("Load Word2vec...")
model = KeyedVectors.load_word2vec_format(
        'datasets/glove.6B.300d.word2vec',
        binary=False)

tokenizer = RegexpTokenizer(r'\w+')

row_index = 0

print("Replacing the words for the word2vec array...")
for row in dataset.itertuples():

    words = tokenizer.tokenize(row[1])

    filtered_words = [word for word in words if word in model.vocab]

    words_count = float32(len(filtered_words))

    for word in filtered_words:

        word_vector = model.word_vec(word)
        x[row_index] = add(x[row_index], word_vector)

    # x[row_index] = divide(x[row_index], words_count)

    row_index += 1

print("Features shape: ", x.shape)
print("Labels shape: ", y.shape)

print("Instantiate the classifier...")
# classifier = naive_bayes.MultinomialNB()
# classifier = RandomForestClassifier(n_estimators=15, n_jobs=-1)
classifier = SVC(C=5, gamma=0.05, kernel="sigmoid", probability=True)
# classifier = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

print("Cross validating...")
scores = cross_val_score(classifier, x, y, cv=5,
                         n_jobs=-1, scoring='neg_log_loss')

end = time.time()
elapsed = end - start

print("Logloss: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Elapsed time: ", elapsed)
