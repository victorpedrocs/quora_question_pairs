import csv

from pandas import read_csv
from numpy import array


def generate(percentage, out_type):
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

    size = dataset.shape[0]
    # sample_ds = dataset.sample(n=int(percentage * size))
    sample_size = int(percentage * size)
    sample_ds = dataset.groupby(1, group_keys=False).apply(lambda x: x.sample(min(len(x), int(sample_size/2))))

    out = open('datasets/dataset_' + out_type + '_' + str(percentage) + '.csv', mode='w')
    csv_writer = csv.writer(out, delimiter='\t')

    for row in sample_ds.itertuples():
        X = row[1]
        y = row[2]
        csv_writer.writerow([X, y])
    out.close()



