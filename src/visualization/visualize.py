"""
Scripts to visualize data and results.
"""

import pickle
import matplotlib.pyplot as plt

from src.data.make_dataset import TextDataset
from src.models.predict_model import initialize_model as init_gen_model, process_prompt
from src.models.metrics import initialize_model as init_clf_model, compute_toxicity


def load_dataset():
    print("Loading from pickle object...")
    df = pickle.load(open('../../data/interim/text_dataset.pkl', 'rb'))

    print(df.data.head())

    print("All done.")
    return df


def load_models():
    print("Initializing the GeDi model...")
    gedi_model = init_gen_model(0)

    print("Initializing the classifier model...")
    clf_model, clf_tokenizer = init_clf_model()

    print("All done.")
    return gedi_model, clf_model, clf_tokenizer


def plot_distribution_by_toxicity_change(dataset):
    plt.hist(dataset.data['toxic_reduction'], bins=20)
    plt.title("Toxicity change distribution")
    plt.xlabel("Toxicity change")
    plt.ylabel("Number of samples")

    print("Saving the plot to figures/toxicity_change_distribution.png...")
    plt.savefig('../../reports/figures/toxicity_change_distribution.png')

    print("All done.")


def plot_toxicity_score_after_model(dataset, gedi_model, clf_model, clf_tokenizer):
    sentences_sample = dataset.data.sample(n=10)

    print("Processing the sentences...")
    model_results = [
        process_prompt(gedi_model, sent, 1)[0]
        for sent in sentences_sample['toxic']
    ]

    print("Computing the toxicity...")
    model_toxicity = compute_toxicity(clf_model, clf_tokenizer, model_results)
    target_toxicity = compute_toxicity(clf_model, clf_tokenizer, sentences_sample['normal'])
    initial_toxiciy = compute_toxicity(clf_model, clf_tokenizer, sentences_sample['toxic'])

    plt.plot(model_toxicity, 'r', label='Model toxicity')
    plt.plot(target_toxicity, 'b', label='Target toxicity')
    plt.plot(initial_toxiciy, 'g', label='Initial toxicity')

    print("Saving the plot to figures/toxicity_change_after_model.png...")
    plt.savefig('../../reports/figures/toxicity_change_after_model.png')

    print("All done.")


def main():
    print('-' * 30 + " Loading the dataset " + '-' * 30)
    dataset = load_dataset()

    print('-' * 30 + " Loading the models " + '-' * 30)
    gedi_model, clf_model, clf_tokenizer = load_models()

    print('-' * 30 + " Plotting the toxicity change by GPT-2 model" + '-' * 30)
    plot_distribution_by_toxicity_change(dataset)

    print('-' * 30 + " Plotting the toxicity change for random samples after translation " + '-' * 30)
    plot_toxicity_score_after_model(dataset, gedi_model, clf_model, clf_tokenizer)


if __name__ == '__main__':
    main()
