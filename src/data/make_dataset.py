"""
Script to download the data and create dataset.
"""

import os
import requests
import argparse

import pandas as pd
from torch.utils.data import Dataset

from transforms import apply_transforms

DATA_URL = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', default='data/raw/filtered_paramnt', help='Directory to download data to.',
                       type=str)
argparser.add_argument('--filename', default='filtered_paranmt.zip', help='Filename of the downloaded data.', type=str)
argparser.add_argument('--unzip_dir', default='data/raw/filtered_paramnt', help='Directory to unzip data to.', type=str)
argparser.add_argument('--unzip', default=False, help='Whether to unzip the downloaded data.',
                       action=argparse.BooleanOptionalAction)
argparser.add_argument('--remove_zip', default=False, help='Whether to remove the downloaded zip file.',
                       action=argparse.BooleanOptionalAction)


class TextDataset(Dataset):
    def __init__(self, data_path, max_sent_len=128):
        self.data_path = data_path
        self.max_len = max_sent_len

        self.raw_data = self._read_data()
        self.toxic_X, self.normal_X, self.toxic_y, self.normal_y = self._prepare_data()

    def _read_data(self):
        df = pd.read_csv(self.data_path, sep='\t', index_col=0)
        return df

    def _prepare_data(self):
        toxic_X, noraml_X = [], []
        toxic_y, normal_y = [], []
        for _, row in self.raw_data.iterrows():
            if row['ref_tox'] > row['trn_tox']:
                toxic_X.append(row['reference'])
                noraml_X.append(row['translation'])
                toxic_y.append(row['ref_tox'])
                normal_y.append(row['trn_tox'])
            else:
                toxic_X.append(row['translation'])
                noraml_X.append(row['reference'])
                toxic_y.append(row['trn_tox'])
                normal_y.append(row['ref_tox'])

        return toxic_X, noraml_X, toxic_y, normal_y

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.toxic_X[idx], self.normal_X[idx], self.toxic_y[idx], self.normal_y[idx]


def create_dataset(data_path, max_sent_len=128):
    print(f'Creating dataset from {data_path}...')
    dataset = TextDataset(data_path, max_sent_len=max_sent_len)
    print('Done.')

    print('Dataset size:', len(dataset))
    print("Data sample:")
    print(dataset.raw_data.sample(n=5))

    print("All done.")
    return dataset


def download_data():
    args = argparser.parse_args()

    data_dir = args.data_dir
    filename = args.filename
    unzip_dir = args.unzip_dir
    unzip = args.unzip
    remove_zip = args.remove_zip

    if os.path.exists(os.path.join(data_dir, filename)):
        print(f'Data already exists in {data_dir}.')
        return os.path.join(data_dir, filename)

    print('Downloading data...')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(f'1. Getting file from {DATA_URL} to {data_dir}...')
    r = requests.get(DATA_URL)
    with open(os.path.join(data_dir, filename), 'wb') as f:
        f.write(r.content)
    print('Done.')

    if unzip:
        print(f'2. Unzipping data from {data_dir} to {unzip_dir}...')
        os.system(f'unzip {os.path.join(data_dir, filename)} -d {unzip_dir}')
        print('Done.')

    if remove_zip:
        print(f'3. Removing zip file from {data_dir}...')
        os.remove(os.path.join(data_dir, filename))
        print('Done.')

    print('All done.')
    return os.path.join(data_dir, filename)


def main():
    print('-' * 30 + " Downloading data " + '-' * 30)
    result_filename = download_data()

    print('-' * 30 + " Creating dataset " + '-' * 30)
    dataset = create_dataset(result_filename)

    print('-' * 30 + " Applying transforms " + '-' * 30)
    apply_transforms(dataset)


if __name__ == '__main__':
    main()
