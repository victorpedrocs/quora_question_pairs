# quora_dataset_builder
from pandas import read_csv
import csv
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer
from contractions import *

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

def generate(out_type):
    # load the dataset into a dataframe
    dataset = read_csv('./datasets/dataset.csv', header='infer', delimiter='\t')

    # drop unecessary columns
    dataset.drop(dataset.columns[[0,1,2]], axis=1, inplace=True)

    out = open('./datasets/dataset_'+out_type+'.csv', mode='w')
    csv_writer = csv.writer(out, delimiter='\t')
    
    # remove non-ascii characters
    dataset.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
        
    for row in dataset.itertuples():
        q1 = row[1].strip() # copy the tuple content to q1
        q2 = row[2].strip() # copy the tuple content to q2
        # Select the type of dataset
        if out_type == 'simple':
            q1 = tokenize(replace_contractions(q1))
            q2 = tokenize(replace_contractions(q2))
        elif out_type == 'stemm':
            q1 = stem(replace_contractions(q1))
            q2 = stem(replace_contractions(q2))

        # Write the questions in the file
        q = q1 + " " + q2
        y = row[3]
        csv_writer.writerow([q,y])
        # q = q2 + " " + q1
        # csv_writer.writerow([q,y])

    out.close()