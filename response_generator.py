import os
from transformers import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from utils import chunks, remove_text_in_brackets, remove_text_in_stars, remove_emojis, remove_noise
from prompt_generator import dialog_listing, dialog_prompt
import re

def clean_response(resp):
    for i in range(0,len(resp)):
        if "Person B's response" in resp[i]:
            resp[i] = resp[i][resp[i].index(":") + len(":"):].strip()
        resp[i] = resp[i].replace("Person B:",'')
        resp[i] = resp[i].replace('\n', ' ')
        resp[i] = resp[i].replace(':)','').strip('#" \'')
        resp[i] = remove_noise(resp[i])
    return resp

class ResponseGenerator:
    def __init__(self, path, device):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.pad_token    = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})
        tokenizer.add_eos_token = False
        tokenizer.add_bos_token = False
        tokenizer.padding_side = 'left'
        self.tokenizer = tokenizer
    
    def generate(self, contexts, mode, temperature, options, tuning = False):
        
        prompts = []
        for context in contexts:
            if mode == 'diverse':
                prompts.append(dialog_prompt(context, options[0], tuning))
            else:
                for option in options:
                    prompts.append(dialog_prompt(context, option, tuning))
        
        prompts = [self.tokenizer.bos_token + p for p in prompts]
        
        model_inputs = self.tokenizer(prompts, return_tensors="pt", add_special_tokens = False,
                                 padding = True, max_length = 10024).to(self.device)
        
        if mode == 'greedy':
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=64, do_sample=False)
        elif mode == 'contrastive':
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=64, penalty_alpha=0.6, top_k=4)
        elif mode == 'sampling':
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=64, temperature = temperature, do_sample=True, top_p=1.0)
        elif mode == 'diverse':
            generated_ids = self.model.generate(**model_inputs, num_beams=len(options), num_beam_groups=len(options), max_new_tokens=64,
                                                diversity_penalty=5.0, num_return_sequences = len(options), do_sample = False)
        
        inputs_length = model_inputs['input_ids'].shape[1]
        generated_ids = generated_ids[:, inputs_length:]
        
        resp = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        resp = clean_response(resp)
        
        resp = list(chunks(resp, len(options)))
        return resp
    
    def generate_listing(self, context, N):
        prompt = self.tokenizer.bos_token + dialog_listing(context, N)
        model_inputs = self.tokenizer([prompt], return_tensors="pt", add_special_tokens = False).to(self.device)
        resp = []
        while(len(resp)) < N:
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)
            inputs_length = model_inputs['input_ids'].shape[1]
            generated_ids = generated_ids[:, inputs_length:]
            resp = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            resp = resp.replace(':\n',': ').split('\n')
            resp = [resp[0]] + [line for line in resp if 'Person B' in line]
            resp = [line for line in resp if len(line) >= 5]
        
        resp = clean_response(resp)
        return resp