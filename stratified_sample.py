from pandas import read_csv
from numpy import array
from sklearn.model_selection import StratifiedShuffleSplit


def generate(size, out_type):
    dataset = None
    if out_type == 'simple':
        dataset = open('datasets/dataset_simple.csv', 'r')
        dataset = read_csv(dataset, header=None, delimiter='\t')
    elif out_type == 'stemm':
        dataset = open('datasets/dataset_concat_stemm.csv', 'r')
        dataset = read_csv(dataset, header=None, delimiter='\t')
    elif out_type == 'tokenize':
        dataset = open('datasets/dataset_concat_unique.csv', 'r')
        dataset = read_csv(dataset, header=None, delimiter='\t')
