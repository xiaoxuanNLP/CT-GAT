import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer

MAX_LEN = 512


class Bart(nn.Module):
    def __init__(self, base, device = None):
        super(Bart, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained(base)
        self.tokenizer = BartTokenizer.from_pretrained(base, add_prefix_space=True)
        self.device = device
        self.max_len = MAX_LEN

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

    def forward(self, input_ids, input_mask, output_ids):
        output = self.model(input_ids=input_ids, attention_mask=input_mask, labels=output_ids)
        loss = output[0]
        return loss

    def generate(self, text, return_num=100, tem=2.1):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', return_length=512, truncation=True, padding=True)

            ids = inputs['input_ids']
            mask = inputs['attention_mask']

            ids = ids.to(self.device)
            mask = mask.to(self.device)

            generate_ids = self.model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=MAX_LEN,
                num_beams=return_num,
                num_return_sequences=return_num,
                temperature=tem,
                do_sample=True,
                # bad_words_ids =
                # top_p = 0.5
                # num_beam_groups = 2
            )

            preds = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generate_ids]

        return preds
