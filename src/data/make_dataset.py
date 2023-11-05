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
argparser.add_argument('--download_url', help='URL to download the data from.', type=str, default=DATA_URL)
argparser.add_argument('--text_dataset_path', default='data/interim/text_dataset.pkl',
                       help='Path to save processed dataset.', type=str)
argparser.add_argument('--filename', default='filtered_paranmt.zip', help='Filename of the downloaded data.', type=str)
argparser.add_argument('--force_download', help="Download the data even if already on disk.", action='store_true')
argparser.add_argument('--with_transforms', help='Whether to apply the transforms.', action='store_true')
argparser.add_argument('--unzip_dir', default='data/raw/filtered_paramnt', help='Directory to unzip data to.', type=str)
argparser.add_argument('--unzip', help='Whether to unzip the downloaded data.',
                       action='store_true')
argparser.add_argument('--remove_zip', help='Whether to remove the downloaded zip file.',
                       action='store_true')


class TextDataset(Dataset):
    def __init__(self, df=None, data_path=None):
        if df is None and data_path is None:
            raise ValueError('Either df or data_path should be provided.')

        if df is not None:
            self.data = df
        else:
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
        data['toxic_score'] = pd.concat(
            [self.raw_data[self.raw_data['ref_tox'] > self.raw_data['trn_tox']]['ref_tox'],
             self.raw_data[self.raw_data['ref_tox'] < self.raw_data['trn_tox']]['trn_tox']])
        data['normal_score'] = pd.concat(
            [self.raw_data[self.raw_data['ref_tox'] > self.raw_data['trn_tox']]['trn_tox'],
             self.raw_data[self.raw_data['ref_tox'] < self.raw_data['trn_tox']]['ref_tox']])

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def split(self, train_size, val_size, test_size):
        assert train_size + val_size + test_size == 1

        train_data = self.data.sample(frac=train_size)
        val_data = self.data.drop(train_data.index).sample(frac=val_size / (1 - train_size))
        test_data = self.data.drop(train_data.index).drop(val_data.index)

        return TextDataset(train_data), TextDataset(val_data), TextDataset(test_data)


def create_dataset(data_path):
    print(f'Creating dataset from {data_path}...')
    dataset = TextDataset(data_path=data_path)
    print('Done.')

    print('Dataset size:', len(dataset))
    print("Data sample:")
    print(dataset.data.sample(n=5))

    print("All done.")
    return dataset


def download_data(data_dir, download_url, text_dataset_path, filename, force_download, unzip_dir, unzip, remove_zip):
    if os.path.exists(os.path.join(data_dir, filename)) and not force_download:
        print(f'1. Data already exists in {data_dir}.')
        print('Done.')
    else:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        print(f'1. Getting file from {download_url} to {data_dir}...')
        r = requests.get(download_url)
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
    print('All done.')


def main(args):
    data_dir = args.data_dir
    text_dataset_path = args.text_dataset_path
    download_url = args.download_url
    force_download = args.force_download
    with_transforms = args.with_transforms
    filename = args.filename
    unzip_dir = args.unzip_dir
    unzip = args.unzip
    remove_zip = args.remove_zip

    print('-' * 30 + " Downloading data " + '-' * 30)
    result_filename, text_dataset_path = download_data(data_dir, download_url,
                                                       text_dataset_path, filename, force_download, unzip_dir,
                                                       unzip, remove_zip)

    print('-' * 30 + " Creating dataset " + '-' * 30)
    dataset = create_dataset(result_filename)

    if with_transforms:
        print('-' * 30 + " Applying text transforms " + '-' * 30)
        from transforms import apply_transforms
        apply_transforms(dataset)

    print('-' * 30 + " Saving text dataset " + '-' * 30)
    save_dataset(dataset, text_dataset_path)


if __name__ == '__main__':
    args = argparser.parse_args()

    main(args)
