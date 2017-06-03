# quora_dataset_builder
from pandas import read_csv
import csv
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer
from contractions import *
from sklearn.externals.joblib import Parallel, delayed

tokenizer = RegexpTokenizer(r'\w+')
stemmer = EnglishStemmer()

def replace_contractions(q):
    q = q.lower()
    for word in q.split(' '):
        q = q.replace(word, contractions[word]) if word in contractions else q
    return q

def tokenize(q):
    return ' '.join(tokenizer.tokenize(q.lower()))

def stem(q):
    return ' '.join([stemmer.stem(word).lower() for word in tokenizer.tokenize(q)])

def all_prep(text):
    return stem(replace_contractions(text.strip()))

def concat_row_prep(r, writer):
    q1 = all_prep(r[1])
    q2 = all_prep(r[2])
    q = q1 + " " + q2
    y = r[3]
    if len(q1) > 7 and len(q2) > 7:
        writer.writerow([q,y])

def read_file(path):
    dataset = read_csv(path, header='infer', delimiter='\t')
    dataset.drop(dataset.columns[[0,1,2]], axis=1, inplace=True)
    dataset.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
    return dataset

def generate_concat(sample_size):
    print('Generating concat file...')
    dataset = read_file('./datasets/dataset.csv')

    out = open('./datasets/dataset_concat.csv', mode='w')
    csv_writer = csv.writer(out, delimiter='\t')

    if sample_size > 0:
        dataset = dataset.sample(sample_size)

    for row in dataset.itertuples():
        concat_row_prep(row, csv_writer)

    out.close()

def generate_one_question_per_line(sample_size):
    print('Generating one question per line file...')
    dataset = read_file('./datasets/dataset.csv')

    out_X = open('./datasets/dataset_singlequestion_X.csv', mode='w')
    out_y = open('./datasets/dataset_singlequestion_y.csv', mode='w')
    writer_X = csv.writer(out_X, delimiter='\t')
    writer_y = csv.writer(out_y, delimiter='\t')

    if sample_size > 0:
        dataset = dataset.sample(sample_size)

    for r in dataset.itertuples():
        if len(r[1]) > 7 and len(r[2]) > 7:
            writer_X.writerow([all_prep(r[1])])
            writer_X.writerow([all_prep(r[2])])
            writer_y.writerow([r[3]])

    out_X.close()
    out_y.close()
