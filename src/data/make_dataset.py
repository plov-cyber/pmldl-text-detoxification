"""
Script to download the data and create dataset.
"""

import os
import pickle

import requests
import argparse

import pandas as pd
from torch.utils.data import Dataset

DATA_URL = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', default='data/raw/filtered_paramnt', help='Directory to download data to.',
                       type=str)
argparser.add_argument('--text_dataset_path', default='data/interim/text_dataset.pkl',
                       help='Path to save processed dataset.', type=str)
argparser.add_argument('--filename', default='filtered_paranmt.zip', help='Filename of the downloaded data.', type=str)
argparser.add_argument('--unzip_dir', default='data/raw/filtered_paramnt', help='Directory to unzip data to.', type=str)
argparser.add_argument('--unzip', default=False, help='Whether to unzip the downloaded data.',
                       action=argparse.BooleanOptionalAction)
argparser.add_argument('--remove_zip', default=False, help='Whether to remove the downloaded zip file.',
                       action=argparse.BooleanOptionalAction)


class TextDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        self.raw_data = self._read_data()
        self.data = self._prepare_data()

    def _read_data(self):
        df = pd.read_csv(self.data_path, sep='\t', index_col=0)
        return df

    def _prepare_data(self):
        data = pd.DataFrame()
        data['toxic'] = pd.concat([self.raw_data[self.raw_data['ref_tox'] > self.raw_data['trn_tox']]['reference'],
                                   self.raw_data[self.raw_data['ref_tox'] < self.raw_data['trn_tox']]['translation']])
        data['normal'] = pd.concat([self.raw_data[self.raw_data['ref_tox'] > self.raw_data['trn_tox']]['translation'],
                                    self.raw_data[self.raw_data['ref_tox'] < self.raw_data['trn_tox']]['reference']])
        data['toxic_reduction'] = abs(self.raw_data['ref_tox'] - self.raw_data['trn_tox'])

        return data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]


def create_dataset(data_path):
    print(f'Creating dataset from {data_path}...')
    dataset = TextDataset(data_path)
    print('Done.')

    print('Dataset size:', len(dataset))
    print("Data sample:")
    print(dataset.data.sample(n=5))

    print("All done.")
    return dataset


def download_data():
    args = argparser.parse_args()

    data_dir = args.data_dir
    text_dataset_path = args.text_dataset_path
    filename = args.filename
    unzip_dir = args.unzip_dir
    unzip = args.unzip
    remove_zip = args.remove_zip

    if os.path.exists(os.path.join(data_dir, filename)):
        print(f'Data already exists in {data_dir}.')
        return os.path.join(data_dir, filename), text_dataset_path

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
    return os.path.join(data_dir, filename), text_dataset_path


def save_dataset(dataset, path):
    if os.path.exists(path):
        dataset_from_disk = pickle.load(open(path, 'rb'))
        if dataset_from_disk == dataset:
            print(f'Dataset already exists in {path} and is up-to-date.')
            return

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    print(f'Pickling the dataset to {path}...')
    pickle.dump(dataset, open(path, 'wb'))
    print('Done.')


def main():
    print('-' * 30 + " Downloading data " + '-' * 30)
    result_filename, text_dataset_path = download_data()

    print('-' * 30 + " Creating dataset " + '-' * 30)
    dataset = create_dataset(result_filename)

    print('-' * 30 + " Applying text transforms " + '-' * 30)
    from transforms import apply_transforms
    apply_transforms(dataset)

    print('-' * 30 + " Saving text dataset " + '-' * 30)
    save_dataset(dataset, text_dataset_path)


if __name__ == '__main__':
    main()
