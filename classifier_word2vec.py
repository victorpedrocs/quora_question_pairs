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
