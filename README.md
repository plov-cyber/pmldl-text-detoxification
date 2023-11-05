# Text Detoxification using GPT-2 based model

**Lev Rekhlov, BS21-DS-02, l.rekhlov@innopolis.university**

## Basic commands

### Install requirements

To install requirements run:

```
conda create --name <env> --file conda_requirements.txt
```

### Download data

The file `src/data/make_dataset.py` contains the script to download the data.

Here is the usage:

```
python src/data/make_dataset.py --filename dataset.zip
```

The default URL downloads
the [Filtered Paramnt dataset](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip)
Available options can be viewed using `--help` flag.

### Transform data

The file `src/data/make_dataset.py` contains the script to transform the data.
To do the standard text transforms(removing punctuation, lowercasing, etc.) run:

```
python src/data/make_dataset.py --with_transforms
```

### Run prediction

The file `src/models/predict_model.py` contains the script to run prediction.
To run prediction on your prompt, use:

```
python src/models/predict_model.py --prompt "Your prompt"
```

You can specify number of outputs using `--output_num` parameter or use `--toxify` flag to toxify your prompt.
