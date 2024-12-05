import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/storage/longmai/trans_cache/'

import json
from response_generator import ResponseGenerator
from utils import ngram_diversity, semantic_diversity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default='mistralai/Mistral-7B-Instruct-v0.2', required=False, type=str)
parser.add_argument("--num_response_per_input", default=5, required=True, type=int)
parser.add_argument("--method", choices=["base", "oto", "otm",'peft'], required=True, type=str)
parser.add_argument("--temperature", default="1.0", required=False, type=float)
parser.add_argument("--decoding", choices=["list","sampling","diverse"], required=True, type=str)
parser.add_argument("--test_file", default='', required=True, type=str)
parser.add_argument("--save_file", default='', required=True, type=str)
args = parser.parse_args()

sts_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sts_model = sts_model.to('cuda')
model = ResponseGenerator(path = args.model_name_or_path, device = 'cuda')
data = [json.loads(line) for line in open(args.test_file)]

for sample in tqdm(data):
    context = sample['context']
    tuning = True if args.method in ['oto','otm','peft'] else False
    
    if args.method == 'peft':
        k_numbers = list(range(1, args.num_response_per_input + 1))
    else:
        k_numbers = [None] * args.num_response_per_input
    
    if args.decoding == 'list':
        sample['preds'] = model.generate_listing(context, args.num_response_per_input)
    else:
        sample['preds'] = model.generate([context], mode = args.decoding, temperature = args.temperature, options = k_numbers, tuning = tuning)[0]

    for line in sample['preds']:
        print(line)

generated_texts = [s['preds'] for s in data]
print("Ngram score: ", ngram_diversity(generated_texts))
print("Sbert diversity score: ", 1 - semantic_diversity(sts_model, generated_texts))

with open(args.save_file,'w') as f:
    f.write('\n'.join([json.dumps(line) for line in data]))