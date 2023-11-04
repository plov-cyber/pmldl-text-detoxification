"""
Script containing the metrics used to evaluate the model.
"""
import argparse
import warnings

warnings.filterwarnings('ignore')

import torch
import transformers
from transformers import RobertaForSequenceClassification, RobertaTokenizer

transformers.logging.set_verbosity_error()

from config import CLF_PATH, DEFAULT_PROMPT, DEVICE

argparser = argparse.ArgumentParser()
argparser.add_argument('--prompt', help="Text prompt to compute toxicity for", type=str)


def initialize_model():
    print("Initializing the model...")
    model = RobertaForSequenceClassification.from_pretrained(CLF_PATH)

    print("Initializing the tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(CLF_PATH)

    print("All done.")
    return model, tokenizer


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

    print('-' * 30 + " Result " + '-' * 30)
    print(f"Prompt: {prompt}")
    print(f"Predicted toxicity: {'%.3f' % toxicity}")


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
