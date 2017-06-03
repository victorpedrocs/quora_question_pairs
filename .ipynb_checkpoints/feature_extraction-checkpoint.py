
from pandas import read_csv
from numpy import load, ones, save, array, add
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from itertools import tee
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import RegexpTokenizer

def pairwise(it):
    a, b = tee(it)
    next(b, None)
    return zip(a, b)

def read_file(input_type):
    '''
        type == 1: Concat questions
        type == 2: Single question per line
    '''
    if input_type == 1:
        path = './datasets/dataset_concat.csv'
        ds = read_csv(path, delimiter='\t', header=None, names=['q', 'y'])
        X = ds.iloc[:,0]
        y = array(ds.iloc[:,1]).ravel()
        return (X, y)

    elif input_type == 2:
        path_X = './datasets/dataset_singlequestion_X.csv'
        path_y = './datasets/dataset_singlequestion_y.csv'
        X = read_csv(path_X, delimiter='\t', header=None, names=['q'])
        y = read_csv(path_y, delimiter='\t', header=None, names=['y'])
        return (X, y)

def save_features(name, X, y):
    save('./npyarrays/X/'+name+'_X', X)
    save('./npyarrays/'+name+'_y', y)

def join_question_pair(X):
    X = [a+b for a, b in pairwise(X)]
    del X[1::2]
    return X

def bag_of_words(input_type=1):
    file_name = 'bigram_bow_svd_300_concat' if input_type == 1 else 'bigram_bow_svd_300_singlequestion'

    print('Generating bag of words...')
    print('File:', file_name)
    N_DIM = 300
    NG_RANGE = (1,2)
    (X, y) = read_file(input_type)
    
    X.head()

    vectorizer = CountVectorizer(ngram_range=NG_RANGE, min_df=1 )
    X = vectorizer.fit_transform(X)
    print(X.shape)

    svd = TruncatedSVD(n_components=N_DIM)
    X = svd.fit_transform(X)

    # if the input is one question per row, concat both together
    if input_type == 2:
        X = join_question_pair(X)
        # SVD again in the 600 feature set
        X = svd.fit_transform(X)

    save_features(file_name, X, y)

def word2vec(input_type=1):
    file_name = 'word2vec_norm_concat' if input_type == 1 else 'word2vec_norm_singlequestion'
    print('Generating word2vec...')
    print('File:', file_name)
    tokenizer = RegexpTokenizer(r'\w+')

    model = KeyedVectors.load_word2vec_format(
            'datasets/glove.6B.300d.word2vec',
            binary=False)
    (X, y) = read_file(input_type)

    row_index = 0
    for row in X:
        words = tokenizer.tokenize(row)
        filtered_words = [word for word in words if word in model.vocab]

        for word in filtered_words:
            word_vector = model.word_vec(word)
            X[row_index] = add(X[row_index], word_vector)
            X[row_index] = divide(X[row_index], norm(X[row_index]))

        row_index += 1

    save_features(file_name, X, y)
