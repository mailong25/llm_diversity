from tqdm import tqdm
from utils import eval_dialog, extract_score, llama_generate, gpt4_generate, chunks
import argparse
from multiprocessing import Pool
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--eval_file", default='', required=True, type=str)
parser.add_argument("--save_file", default='', required=True, type=str)
args = parser.parse_args()

EVAL_FUNC = eval_dialog
fast_eval_score = False
metrics = ['Coherence rating:']
CHUNK_SIZE = 64

data = [json.loads(line) for line in open(args.eval_file)][:60]

contexts = [line['context'] for line in data]
preds = [line['preds'] for line in data]

for i in range(0,len(contexts)):
    contexts[i] = [contexts[i]] * len(preds[i])
contexts = [j for i in contexts for j in i]
preds = [j for i in preds for j in i]

prompts, logs = [], []

for context, next_resp in zip(contexts, preds):
    prompts.append(EVAL_FUNC(context, next_resp))
    logs.append({'context': context, 'response': next_resp, 'prompt' : prompts[-1],
                 'chat_out': None, 'chat_score': None, 'gpt4_out': None, 'gpt4_score': None})

chunk_prompts = list(chunks(prompts, CHUNK_SIZE))
pool = Pool(16)

c_idx = 0
for chunk in tqdm(chunk_prompts):
    outs = pool.map(llama_generate, chunk)
    scores = [extract_score(out, metrics, fast = False) for out in outs]
    error_ids = []
    
    for i in range(0,len(scores)):
        logs[c_idx*CHUNK_SIZE + i]['chat_out'] = outs[i]
        logs[c_idx*CHUNK_SIZE + i]['chat_score'] = scores[i]
        
        if scores[i]['Coherence rating:'] is None or scores[i]['Coherence rating:'] < 6:
            error_ids.append(c_idx*CHUNK_SIZE + i)
    
    error_prompts = [EVAL_FUNC(logs[idx]['context'], logs[idx]['response']) for idx in error_ids]
    error_outs = pool.map(gpt4_generate, error_prompts)
    
    for error_idx, error_out in zip(error_ids, error_outs):
        flat_idx = error_idx
        logs[flat_idx]['gpt4_out'] = error_out
        logs[flat_idx]['gpt4_score'] = extract_score(error_out, metrics, fast = False)
    c_idx += 1

pickle.dump(logs, open(args.save_file, "wb"))

scores = [line['gpt4_score'] if line['gpt4_score'] != None else line['chat_score'] for line in logs]
count = 0
for i in range(0,len(scores)):
    if scores[i]['Coherence rating:'] <= 5:
        count += 1
print("Incoherence rate:", round((count / len(scores))* 100, 1))