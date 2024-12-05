import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/storage/longmai/trans_cache/'

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging as hf_logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from datasets import load_dataset
from trl import SFTTrainer
import bitsandbytes as bnb
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import argparse
import math
import json
import random
import sys
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from accelerate.utils import set_seed
from utils import add_personAB

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune LLMs")
    parser.add_argument("--train_file", type=str, default='path to train_file',)
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--ul_weight", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=512,)
    parser.add_argument("--max_input_length", type=int, default=250)
    parser.add_argument("--clip_norm", action="store_true", help="")
    parser.add_argument("--max_pos_number", type=int, default=9)
    parser.add_argument("--max_steps", type=int, default=1000,)
    parser.add_argument("--num_response_per_input", type=int, default=4,)
    parser.add_argument("--no_ul", action="store_true", help="")
    parser.add_argument("--max_num_negatives_per_sample", type=int, default=50)
    parser.add_argument("--skip_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument(
        "--loss_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default = 'mistralai/Mistral-7B-Instruct-v0.2',
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        required = True,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--output_dir", type=str, default='cons_res2', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

logger = logging.getLogger(__name__)
def main():
    args = parse_args()
    accelerator = Accelerator()
    
    logging.basicConfig(filename = os.path.join(args.output_dir, args.exp_name + '_log.txt'),
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info(str(args))
    logger.info(accelerator.state)
    logger.info(accelerator.device)
    
    set_seed(args.seed)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})
    tokenizer.add_eos_token = False
    tokenizer.add_bos_token = False
    tokenizer.padding_side = 'left'
    
    model = prepare_model_for_kbit_training(model)    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if 'llama' in args.model_name:
        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            bias="none",
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )
    else: 
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
    
    model = get_peft_model(model, peft_config)
    
    def prompt_dialog(dialog_context, option):
        context = add_personAB(dialog_context,  args.max_input_length)
        if args.no_ul:
            task = "Imagine you are person B and act as if you were a real individual. Please write the next response of person B."
        else:
            task = "Imagine you are person B and act as if you were a real individual. Think about all the possibilities in which person B might respond next and then provide the response that corresponds to possibility number #" + str(option) + '.'
        prompt  = "Given this conversation:\n\n"
        prompt += '\n'.join(context) + '\n\n' + task
        return prompt
    
    START_INST = '[INST] '
    END_INST = ' [/INST] '
    
    skip_toks = stopwords.words('english') + list(string.punctuation)    
    prompt_end_tok = tokenizer.encode(END_INST.strip(' '), add_special_tokens=False)
    
    def find_sublist(sublist, mainlist):
        len_sub = len(sublist)
        for i in range(len(mainlist)-len_sub+1):
            if mainlist[i:i+len_sub] == sublist:
                return i
        return None
    
    def standardize_text(text):
        return re.sub(r'[^\w\s]', '', text).lower()
    
    def remove_lead_punct(s):
        return re.sub(r'^[^\w]+', '', s)
    
    def create_batch(samples):
        sample = samples[0]
        assert len(sample['candidates']) >= args.num_response_per_input
        sample['candidates'] = sample['candidates'][:args.num_response_per_input]
        
        common_words = set(skip_toks)
        prompts = []
        
        promt_func = prompt_dialog
        
        idxs = list(range(0, args.max_pos_number))
        idxs = random.sample(idxs, len(sample['candidates']))
        
        for candidate in sample['candidates']:
            idx = idxs[sample['candidates'].index(candidate)]
            inp = promt_func(sample['context'], idx + 1)
            text  = tokenizer.bos_token + START_INST + inp + END_INST
            text += ' '.join((candidate + ' ' + tokenizer.eos_token).split()[:60])
            prompts.append(text)
        
        if args.no_ul is False:            
            neg_prompts = []
            for i in range(0,len(sample['candidates'])):
                idx = idxs[i]
                for j in range(0,len(sample['candidates'])):
                    if i != j:
                        inp = promt_func(sample['context'], idx + 1)
                        text  = tokenizer.bos_token + START_INST + inp + END_INST
                        text += ' '.join((sample['candidates'][j] + ' ' + tokenizer.eos_token).split()[:60])
                        neg_prompts.append(text)
            random.shuffle(neg_prompts)
            neg_prompts = neg_prompts[:args.max_num_negatives_per_sample]
            prompts += neg_prompts
        
        inputs  = tokenizer(prompts, truncation=True, max_length=args.max_length, return_tensors = 'pt',
                            padding=True, add_special_tokens=False,)
        
        prompt_ends, ul_labels, attention_masks = [], [], []
        labels = inputs['input_ids'].tolist()
        
        for j in range(0,len(labels)):
            prompt_end_idx = find_sublist(prompt_end_tok, labels[j]) + len(prompt_end_tok) - 1
            
            prompt_toks = labels[j][:prompt_end_idx].copy()
            out_toks    = labels[j][prompt_end_idx:].copy()
            
            labels[j][:prompt_end_idx] = [-100] * len(labels[j][:prompt_end_idx])
            
            lead_toks = prompts[j].split(END_INST)[1].split()[:-1]
            lead_toks = [remove_lead_punct(tok) for tok in lead_toks if tok not in common_words and standardize_text(tok) not in common_words]
            lead_toks = [tok for tok in lead_toks if tok != '']
            
            if len(lead_toks) > 0:
                tokenized_toks = tokenizer(lead_toks, add_special_tokens=False)['input_ids']
                lead_toks = [tok[1] if word[0].isdigit() else tok[0] for word, tok in zip(lead_toks, tokenized_toks)]
            
            ul_toks = [tok if tok in lead_toks else -100 for tok in out_toks]
            prompt_ends.append(prompt_end_idx)
            ul_labels.append([-100]*len(prompt_toks) + ul_toks)
        
        batch={
            "input_ids": inputs['input_ids'],
            "labels": torch.tensor(labels),
            "ul_labels": torch.tensor(ul_labels),
            "attention_mask": inputs['attention_mask'],
            "prompt_ends": torch.tensor(prompt_ends),
        }        
        return batch
        
    class DialogDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                "context": item['context'],
                "candidates": item['candidates']
            }
    
    train_data = [json.loads(line) for line in open(args.train_file)]    
    train_dataset = DialogDataset(train_data)    
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=False, collate_fn = create_batch)
    
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = bnb.optim.AdamW32bit(optimizer_grouped_parameters, lr= args.learning_rate, is_paged = True)
    model,optimizer,train_dataloader = accelerator.prepare(model,optimizer,train_dataloader)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch 
    
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    combine_loss = []
    combine_uls = []
    
    model.train()
    for epoch in range(args.num_train_epochs):
        random.seed(42)
        for step, batch in enumerate(train_dataloader):
            if completed_steps < args.skip_steps:
                completed_steps += 1
                progress_bar.update(1)
                continue
            
            num_sample = args.num_response_per_input
            
            outputs = model(input_ids = batch['input_ids'],
                            attention_mask = batch['attention_mask'],
                            labels = batch['labels'],
                            return_dict = True, output_hidden_states = True)
            logits = outputs.logits
            
            mle_logits = logits[:num_sample].float()
            labels = batch['labels'][:num_sample]
            shift_logits = mle_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            mle_loss = loss_fct(shift_logits, shift_labels)
            
            if args.no_ul is False and completed_steps >= args.warmup_steps:
                ul_logits = logits[num_sample:]
                ul_logits = ul_logits[:,:-2,:].float()
                labels = batch['ul_labels'][num_sample:]
                labels = labels[:,:-2]
                
                shift_logits = ul_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                
                output_ul = F.softmax(shift_logits, dim = -1)
                scores = 1.0 - output_ul
                scores = torch.log(torch.clamp(scores, min=1e-6))
                ul_loss = F.nll_loss(scores, shift_labels).float()
                loss = mle_loss + args.ul_weight*ul_loss
            else:        
                ul_loss = mle_loss
                loss = mle_loss
            
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            
            completed_steps += 1
            combine_loss.append(mle_loss.detach().item())
            combine_uls.append(ul_loss.detach().item())
            
            if completed_steps % args.gradient_accumulation_steps == 0:
                if args.clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            
            if completed_steps % args.loss_steps == 0 and len(combine_loss) > 0:
                avg_loss = sum(combine_loss) / len(combine_loss)
                logger.info("MLE Loss:" + str(avg_loss))
                if len(combine_uls) > 0:
                    avg_uls = sum(combine_uls) / len(combine_uls)
                    logger.info("UL Loss:" + str(avg_uls))
                combine_loss = []
                combine_uls = []
            
            if completed_steps % 10 == 0:
                torch.cuda.empty_cache()
            
            if completed_steps % args.save_steps != 0:
                continue
            
            model.eval()
            checkpoint_dir = os.path.join(args.output_dir, args.exp_name, 'check_' + str(completed_steps))
            os.makedirs(checkpoint_dir, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(checkpoint_dir)
            model.train()
            
            if completed_steps >= args.max_steps:
                sys.exit(0)
            
if __name__ == "__main__":
    main()