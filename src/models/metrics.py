"""
Script containing the metrics used to evaluate the model.
"""
import argparse
import warnings

warnings.filterwarnings('ignore')

import torch
import transformers
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from nltk.translate.bleu_score import sentence_bleu

transformers.logging.set_verbosity_error()

from config import CLF_PATH, DEFAULT_PROMPT, DEVICE
from predict_model import initialize_model as init_gen_model, process_prompt

argparser = argparse.ArgumentParser()
argparser.add_argument('--prompt', help="Text prompt to compute toxicity for", type=str)


def initialize_model():
    print("Initializing the model...")
    model = RobertaForSequenceClassification.from_pretrained(CLF_PATH)

    print("Initializing the tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(CLF_PATH)

    print("All done.")
    return model, tokenizer


def get_metrics(prompt, result):
    metrics = dict()

    model, tokenizer = initialize_model()

    print("Computing metrics...")
    metrics['bleu'] = sentence_bleu([prompt], result)
    metrics['toxicity'] = compute_toxicity(model, tokenizer, [prompt])[0]

    print("All done.")
    return metrics


def comput_blue_score(prompt):
    model = init_gen_model(0)
    result = process_prompt(model, prompt, 1)[0]

    print('Calculating BLEU similarity...')
    bleu_sim = sentence_bleu([prompt], result)

    print("All done.")
    return result, float(bleu_sim)


def compute_toxicity(model, tokenizer, prompts):
    with torch.inference_mode():
        print("Encoding the prompt...")
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(DEVICE)

        print("Computing the toxicity...")
        out = torch.softmax(model(**inputs).logits, -1)[:, 1].cpu().numpy()

    print("All done.")
    return out


def main(args):
    prompt = args.prompt

    print('-' * 30 + " Initializing the model " + '-' * 30)
    model, tokenizer = initialize_model()

    print('-' * 30 + " Computing toxicity " + '-' * 30)
    if prompt is None:
        print(f"!!! You didn't specify the prompt, using default: '{DEFAULT_PROMPT}' !!!")
        prompt = DEFAULT_PROMPT

    toxicity = compute_toxicity(model, tokenizer, [prompt])[0]

    print('-' * 30 + " Computing BLEU score " + '-' * 30)
    model_result, bleu_score = comput_blue_score(prompt)

    print('-' * 30 + " Result " + '-' * 30)
    print(f"Prompt: {prompt}")
    print(f"Translation: {model_result}")
    print(f"BLEU score:\t{'%.4f' % bleu_score}")
    print(f"Toxicity:\t{'%.4f' % toxicity}")


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
