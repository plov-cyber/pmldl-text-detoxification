"""
Script with the model class.
"""
import numpy as np
import torch

from transformers.generation_utils import GenerationMixin


class GediAdapter(GenerationMixin):
    def __init__(
            self, model, gedi_model,
            target=0,
            gedi_logit_coef=1,
            pos_code=1349,
            neg_code=13086,
            tokenizer=None,
            max_id=None,
            debug=False,
            reg_alpha=0,
            ub=None,
            lb=None,
            untouchable_tokens=None,
            nearly_infinity=-1000,
    ):
        self.model = model
        self.gedi_model = gedi_model
        self.target = target
        self.gedi_logit_coef = gedi_logit_coef
        self.POS_CODE = pos_code
        self.NEG_CODE = neg_code
        self.codes = {'gedi_pos': self.POS_CODE, 'gedi_neg': self.NEG_CODE}
        self.tokenizer = tokenizer
        self.max_id = max_id
        self.debug = debug
        self.reg_alpha = reg_alpha
        self.ub = ub
        self.lb = lb
        self.logits = []
        self.untouchable_tokens = untouchable_tokens or []
        self.nearly_infinity = nearly_infinity

    def show_correction(self, sm, logits, corrected, old_logits):
        if self.tokenizer:
            vals = sm.cpu().numpy()[0][0]
            lv = logits.cpu().numpy()[0]
            cv = corrected.cpu().numpy()[0]
            if self.max_id:
                vals = vals[:self.max_id]
                lv = lv[:self.max_id]
                cv = cv[:self.max_id]
            # the most upgraded and downgraded tokens
            print('+', self.tokenizer.convert_ids_to_tokens(np.argsort(-vals)[:5]), -np.sort(-vals)[:3])
            print('-', self.tokenizer.convert_ids_to_tokens(np.argsort(vals)[:5]), np.sort(vals)[:3])
            print(torch.exp(logits).sum())
            # how the top logits change
            old_top_id = np.argsort(-lv)[:5]
            new_top_id = np.argsort(-cv)[:5]
            toks = list(new_top_id)
            for t in old_top_id:
                if t not in toks:
                    toks.append(t)

            pos_logits = old_logits[0][0].cpu().numpy()
            neg_logits = old_logits[1][0].cpu().numpy()

            texts = self.tokenizer.convert_ids_to_tokens(toks)
            print('changes in the top:')
            for text, idx in zip(texts, toks):
                print('{:6d}: {:+2.2f} > {:+2.2f} {:20s}     [{:+2.2f} | {:+2.2f}]'.format(idx, lv[idx], cv[idx], text,
                                                                                           pos_logits[idx],
                                                                                           neg_logits[idx]))
            print()
            print(self.tokenizer.convert_ids_to_tokens(old_top_id), self.tokenizer.convert_ids_to_tokens(new_top_id))

    def __call__(self, return_dict=True, **kwargs):
        new_args = kwargs.get('main', {})
        with torch.no_grad():
            outputs = self.model(return_dict=return_dict, **new_args)
        outputs['main'] = outputs
        gedi_logits = {}
        for gedi_key in ['gedi_pos', 'gedi_neg']:
            gedi_args = kwargs.get(gedi_key, {})
            with torch.no_grad():
                gedi_out = self.gedi_model(**gedi_args, return_dict=True)
            outputs[gedi_key] = gedi_out  # logits are [batch, seq, voc]
            gedi_logits[gedi_key] = gedi_out['logits'][:, -1]  # [batch, voc]

        stacked = torch.stack([gedi_logits['gedi_pos'], gedi_logits['gedi_neg']])  # [2, batch, voc]
        # exclude untouchable tokens from the distribution on which the penalty is calculated
        for token_id in self.untouchable_tokens:
            stacked[:, :, token_id] = self.nearly_infinity
        if self.reg_alpha:
            # increase each p(token|class) by the same amount, to shift odds ratio to 1.
            old_logits = torch.log(torch.softmax(stacked, -1) + self.reg_alpha)
        else:
            old_logits = torch.log_softmax(stacked, -1)

        if hasattr(self.gedi_model, 'logit_scale'):
            old_logits += self.gedi_model.logit_scale
        if hasattr(self.gedi_model, 'bias'):
            old_logits += self.gedi_model.bias.reshape(2, 1, 1).repeat(1, 1, old_logits.shape[-1])

        sm = torch.log_softmax(old_logits, 0)
        logits = outputs['logits'][:, -1]

        shift = sm[self.target]
        # shift everything by a constant to make logits before and after change more comparable
        shift -= shift.mean()
        # limit the positive or negative impact of gedi correction
        if self.lb is not None or self.ub is not None:
            shift = torch.clamp(shift, self.lb, self.ub)
        for token_id in self.untouchable_tokens:
            shift[:, token_id] = 0

        corrected = logits + shift * self.gedi_logit_coef
        if self.debug:
            self.show_correction(sm, logits, corrected, torch.log_softmax(stacked, -1))
        if self.max_id is not None:
            corrected[self.max_id:] = -np.infty
        outputs['logits'] = corrected.unsqueeze(1)  # add back sequence length

        return outputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        result = {}
        # unpack past after beam search application
        past = kwargs.get('past')
        if past and not isinstance(past, tuple):
            for k, v in past.items():
                kwargs[k]['past'] = v

        main_input_ids = input_ids
        main_kwargs = kwargs.get('main', kwargs)
        if kwargs.get('main_prefix') is not None and main_kwargs.get('past') is None:
            prefix = kwargs['main_prefix'].unsqueeze(0).repeat(main_input_ids.shape[0], 1)
            main_input_ids = torch.cat([prefix, main_input_ids], dim=1)
            if main_kwargs.get('attention_mask') is not None:
                old_mask = main_kwargs['attention_mask']
                mask_prefix = prefix * 0 + 1
                main_kwargs['attention_mask'] = torch.cat([mask_prefix, old_mask], dim=1)
        result['main'] = self.model.prepare_inputs_for_generation(main_input_ids, **main_kwargs)

        for k in ['gedi_pos', 'gedi_neg']:
            gedi_args = kwargs.get(k, {})
            if kwargs.get('gedi_prepend'):
                # prepend the code to the input
                prefix = torch.ones([input_ids.shape[0], 1], dtype=input_ids.dtype).to(input_ids.device) * self.codes[k]
                new_input_ids = torch.cat([prefix, input_ids], dim=1)
            else:
                # instert the code instead of the first token of the input
                new_input_ids = input_ids.clone()  # batch size x seq len
                new_input_ids[:, 0] = self.codes[k]
            gedi_inputs = self.gedi_model.prepare_inputs_for_generation(new_input_ids, **gedi_args)
            result[k] = gedi_inputs
        return result

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        result = {k: v for k, v in model_kwargs.items()}
        result['main'] = self.model._update_model_kwargs_for_generation(
            outputs=outputs['main'],
            model_kwargs=model_kwargs.get('main', model_kwargs),
            is_encoder_decoder=self.model.config.is_encoder_decoder,
        )

        for k in ['gedi_pos', 'gedi_neg']:
            result[k] = self.gedi_model._update_model_kwargs_for_generation(
                outputs=outputs[k],
                model_kwargs=model_kwargs.get(k, {}),
                is_encoder_decoder=self.gedi_model.config.is_encoder_decoder,
            )

        # a fix for beam search
        result['past'] = {
            k: result[k]['past']
            for k in ['main', 'gedi_pos', 'gedi_neg']
            if 'past' in result[k] and result[k]['past'] is not None and result[k]['past'][0] is not None
        }
        return result

    def _reorder_cache(self, past, beam_idx):
        # for each model, cache should be reordered separately
        result = {}
        for key, subpast in past.items():
            model = self.model if key == 'main' else self.gedi_model
            result[key] = model._reorder_cache(subpast, beam_idx)
        return result

    @property
    def config(self):
        return self.model.config

    def get_encoder(self):
        return self.model.get_encoder()

    def parameters(self):
        return self.model.parameters()

    @property
    def device(self):
        return self.model.device

    @property
    def main_input_name(self):
        return self.model.main_input_name

    def forward(self, attention_mask=None, **kwargs):
        pass
