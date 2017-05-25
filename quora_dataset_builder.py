from pandas import read_csv
import csv
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer

dataset = open('datasets/dataset.csv', 'r')
dataset = read_csv(dataset, header='infer', delimiter='\t')

tokenizer = RegexpTokenizer(r'\w+')
stemmer = EnglishStemmer()
#stopwords_set = set(stopwords.words('english'))

def pre_process_question(q):
    return ' '.join(set(tokenizer.tokenize(q)))
    # return ' '.join([stemmer.stem(word) for word in tokenizer.tokenize(q)])
    #return ' '.join([stemmer.stem(word) for word in tokenizer.tokenize(q) if word not in stopwords_set])


#dataset['q1'].apply(pre_process_question)
#dataset['q2'].apply(pre_process_question)

out = open('dataset_concat.csv', mode='w')
csv_writer = csv.writer(out, delimiter='\t')

for row in dataset.itertuples():
    # q1 = pre_process_question(row[4]).strip()
    # q2 = pre_process_question(row[5]).strip()
    # q = q1 + " " + q2
    q1 = row[4].strip()
    q2 = row[5].strip()
    q = q1 + " " + q2
    y = row[6]
    csv_writer.writerow([q,y])

out.close()
