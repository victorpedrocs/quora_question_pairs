#%%
from pandas import read_csv
import csv
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer
from contractions import *
from sklearn.model_selection import StratifiedShuffleSplit

def replace_contractions(q):
    q = q.lower()
    for word in q.split(' '):
        q = q.replace(word, contractions[word]) if word in contractions else q
    return q

def tokenize(q):
    return ' '.join(set(tokenizer.tokenize(q.lower())))

def stem(q):
    return ' '.join([stemmer.stem(word).lower() for word in tokenizer.tokenize(q)])

def generate(out_type):
    # open the raw dataset
    dataset = open('datasets/dataset.csv', 'r')
    # load into a dataframe
    dataset = read_csv(dataset, header='infer', delimiter='\t')
    # instantiate tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # instantiate stemmer
    stemmer = EnglishStemmer()
    # drop unecessary columns
    dataset.drop(dataset.columns[[0,1,2]], axis=1, inplace=True)

    # limit to the size
    if out_type == 'simple':
        dataset['q1'] = dataset['q1'].apply(replace_contractions)
        dataset['q2'] = dataset['q2'].apply(replace_contractions)
    elif out_type == 'tokenize':
        dataset['q1'] = dataset['q1'].apply(tokenize)
        dataset['q2'] = dataset['q2'].apply(tokenize)
    elif out_type == 'stemm':
        dataset['q1'] = dataset['q1'].apply(stem)
        dataset['q2'] = dataset['q2'].apply(stem)

    out = open('datasets/dataset_'+out_type+'.csv', mode='w')
    csv_writer = csv.writer(out, delimiter='\t')

    for row in dataset.itertuples():
        q1 = row[1].strip() # copy the tuple content to q1
        q2 = row[2].strip() # copy the tuple content to q2
        q = q1 + " " + q2
        y = row[3]
        csv_writer.writerow([q,y])
        q = q2 + " " + q1
        csv_writer.writerow([q,y])

    out.close()
