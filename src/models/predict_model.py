"""
Script to make predictions using the trained model
"""
import argparse
import warnings

warnings.filterwarnings('ignore')

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

transformers.logging.set_verbosity_error()

from model_class import GediAdapter
from config import TOKENIZER_PATH, MODEL_PATH, DIS_PATH, DEVICE, DEFAULT_PROMPT, \
    TOXIC_MODEL_CONF, NONTOXIC_MODEL_CONF, COMMON_GEN_CONF

argparser = argparse.ArgumentParser()
argparser.add_argument('--prompt', help="Text prompt to detoxify by the model", type=str)
argparser.add_argument('--output_num', default=1, help='Number of model ouputs', type=int)
argparser.add_argument('--toxify', help="Make the sentence toxic instead", action='store_true')


def initialize_model(toxify):
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("Initializing generator...")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    gen_model.resize_token_embeddings(len(tokenizer))

    print("Initializing discriminator...")
    dis_model = AutoModelForCausalLM.from_pretrained(DIS_PATH)

    new_pos = tokenizer.encode('normal', add_special_tokens=False)[0]
    new_neg = tokenizer.encode('toxic', add_special_tokens=False)[0]

    dis_model.bias = torch.tensor([[0.08441592, -0.08441573]])
    dis_model.logit_scale = torch.tensor([[1.2701858]])

    print("Combining all together...")
    gedi_adapter = GediAdapter(
        model=gen_model,
        gedi_model=dis_model,
        tokenizer=tokenizer,
        neg_code=new_neg,
        pos_code=new_pos,
        **(TOXIC_MODEL_CONF if toxify else NONTOXIC_MODEL_CONF)
    )

    print("All done.")
    return gedi_adapter


def process_prompt(model, prompt, output_num):
    print("Encoding the prompt...")

    inputs = model.tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

    print("Generating the output...")
    results = model.generate(
        inputs,
        num_return_sequences=output_num,
        **COMMON_GEN_CONF,
    )

    print("Decoding the results...")
    results = [
        model.tokenizer.decode(r, skip_special_tokens=True)
        for r in results
    ]

    print("All done.")
    return results


def print_prompt_results(prompt, results):
    print("Your prompt:", prompt)
    print("Variations of results:")
    for i, r in enumerate(results):
        print(f"{i + 1}. {r}")


def main(args):
    prompt = args.prompt
    toxify = args.toxify
    output_num = args.output_num

    print('-' * 30 + " Initializing the model " + '-' * 30)
    gedi_adapter = initialize_model(toxify)

    print('-' * 30 + " Processing the prompt " + '-' * 30)
    if prompt is None:
        print(f"!!! You didn't specify the prompt, using default: '{DEFAULT_PROMPT}' !!!")
        prompt = DEFAULT_PROMPT

    results = process_prompt(gedi_adapter, prompt, output_num)

    print('-' * 30 + " Result " + '-' * 30)
    print_prompt_results(prompt, results)


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args)
