# Solution Building Report

## 1. Dataset Exploration

The dataset contains 577.777 rows with two sentences in each: **toxic** and **neutral** versions.

As the helping information we have:

- **toxicity level** for each version
- **relative difference of lengths** between versions
- **cosine similarity** of the versions

I preliminarily chose the level of toxicity as the metric for further model evaluation.

## 2. Data Preprocessing

Before researching any information about models, I decided to write some code for text cleaning.

I used the standard preprocessing techniques:

- lowercasing
- removing punctuation
- removing stopwords
- lemmatization
- tokenization

During this phase, I tried to derive **the set of toxic words** from the dataset.
My idea was the following: let's find the set difference between all words from toxic sentences and all words from
neutral sentences.
The resulting set will contain only toxic words.

However, I expected, that the sentences are still pretty dirty even after preprocessing.

## 3. Toxicity Classifier

Why not to try the simplest pipeline for NLP?
From scikit-learn I imported the following:

- TfidfVectorizer
- GaussianNB

I used the vectorizer to transform the sentences into vectors of TF-IDF features.
Then I trained the classifier on the resulting vectors.

Even such simple pipeline showed the accuracy of 0.68 on the test set. Here are some more metrics:

**Precision**: 0.77

**Recall**: 0.61

**F1-Score**: 0.68

However, I still won't use this as a final toxicity classifier.

## 4. Choosing the model

The main paper for this assignment was very useful. To keep things simple and not invent the bicycle, I decided to take
the model from the paper and write convenient API for it.

The model I chose is **GeDi** from the paper. Under the hood, it is GPT-2 model, used for generation and also T5
paraphraser, fine-tuned to detoxify the words.

The model is available on HuggingFace, so I used it as a base for my solution.

## 5. Exploring other models

I also tried to use another technique: **removing toxic words** from the sentence and checking the toxicity.

As toxicity classifier, I also used model from paper: RoBERTa fine-tuned for toxicity classification.

From the notebook, you can see, that this approach is not very good. Even on examples we can note many mistakes:

- many sentences left the same
- some sentences lose their meaning

## 6. Final solution

So in the end, we have the following models:

- GeDi model for paraphrasing
- RoBERTa model for toxicity classification