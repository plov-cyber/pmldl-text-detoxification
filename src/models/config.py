TOKENIZER_PATH = "s-nlp/t5-paraphrase-paws-msrp-opinosis-paranmt"
MODEL_PATH = TOKENIZER_PATH
DIS_PATH = 's-nlp/gpt2-base-gedi-detoxification'
DEVICE = "cpu"
DEFAULT_PROMPT = "Got damn, you didn't specify any prompt, shit!"

COMMON_GEN_CONF = {
    # 'do_sample': True,
    'num_beams': 5,
    'repetition_penalty': 5.0,
    'temperature': 1.0
}

TOXIC_MODEL_CONF = {
    'gedi_logit_coef': 5,
    'target': 1
}

NONTOXIC_MODEL_CONF = {
    'gedi_logit_coef': 10,
    'target': 0,
    'reg_alpha': 3e-5,
    'ub': 0.01
}
