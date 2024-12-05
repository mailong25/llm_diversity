OPENAI_KEY = ''
TOGETHER_KEY = ""

from nltk import word_tokenize
import string
import os
import re
import random
import timeout_decorator
import sys
from openai import OpenAI
from sentence_transformers import util

client = OpenAI(api_key = OPENAI_KEY)
punct_set = set([c for c in string.punctuation]) | set(['“','”',"...","–","…","..","•",'“','”'])

def generateNgram(paper, ngram = 2, deli = '_', rmSet = punct_set):
    words = word_tokenize(paper)
    words = [w.lower() for w in words]
    if len(words) == 1:
        return ''
    ngrams = []
    for i in range(0,len(words) - ngram + 1):
        block = words[i:i + ngram]
        if not any(w in rmSet for w in block):
            ngrams.append(deli.join(block))
    return ngrams

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def ngram_diversity(collections):
    uni_gram = [generateNgram(j, ngram = 1) for i in collections for j in i]
    bi_gram  = [generateNgram(j, ngram = 2) for i in collections for j in i]
    uni_gram = [j for i in uni_gram for j in i]
    bi_gram  = [j for i in bi_gram for j in i]
    return {'unigram': round(len(set(uni_gram)) / len(uni_gram), 4),
            'bigram': round(len(set(bi_gram)) / len(bi_gram), 4)}

def semantic_diversity(model, collections):
    if len(collections[0]) == 1:
        return 1
    all_sims = []
    for sentences in collections:
        embeddings = model.encode(sentences)
        sims = util.cos_sim(embeddings, embeddings).tolist()
        for i in range(0,len(sims) - 1):
            all_sims += sims[i][i+1:]
    return round(sum(all_sims) / len(all_sims), 4)

def semantic_coherence(model, contexts, responses):    
    all_sims = []
    for i in range(0,len(contexts)):
        if args.dialog:
            context_embeddings  = model.encode(contexts[i].split('\n')[-1:])
        else:
            context_embeddings  = model.encode(contexts[i])
        response_embeddings = model.encode(responses[i])
        sims = util.cos_sim(context_embeddings, response_embeddings).tolist()[0]
        all_sims += sims
    return round(sum(all_sims) / len(all_sims), 4)

def add_personAB(dialog_context, max_words = 1000, orders = ['Person A: ', 'Person B: ']):
    ''' Dialog context = '\n'.join()'''
    if type(dialog_context) != list:
        dialog_context = dialog_context.split('\n')
    context = list(reversed(dialog_context))
    
    for i in range(0,len(context)):
        if i % 2 == 0:
            context[i] = orders[0] + context[i]
        else:
            context[i] = orders[1] + context[i]
        
        tokens = word_tokenize('\n'.join(context[:i+1]))
        if len(tokens) > max_words:
            break
    
    if len(tokens) < max_words:
        context = context[:i+1]
    else:
        context = context[:i]
    
    context = list(reversed(context))
    return '\n'.join(context)

def eval_dialog(context, response):
    prompt = "Given this dialog:\n\n"    
    prompt += add_personAB(context) + '\n\n'
    prompt += "Does this next response from Person B make coherent sense?\n"
    prompt += '"Person B: ' + response + '"' + '\n\n'
    prompt += "Begin your evaluation by providing a short assessment. Then, rate the coherence of Person B's response on a scale from 1 to 10 by strictly following this example format: 'Coherence rating: [5]'\n\n"
    prompt += 'Coherence assessment:'
    return prompt

def replace_non_ascii(list_string):
    return [remove_noise(s) for s in list_string]

def extract_numbers(input_string):
    numbers = re.findall(r'\d+', input_string)
    return [num for num in numbers]

def contains_number(s):
    return bool(re.search(r'\d', s))

def extract_score(res_string, score_names, fast = False):
    if fast:
        return {score_names[0]: float(extract_numbers(res_string)[0])}
    res_string = res_string.replace('\n\n','\n')
    lines = res_string.split('\n')
    res = {}
    for name in score_names:
        for i in range(0,len(lines)):
            if name in lines[i] and contains_number(lines[i].split(name)[1]) and lines[i].count(':') == 1:
                if name == lines[i]:
                    score = lines[i+1]
                else:
                    score = lines[i].split(name)[1]
                
                score = score.replace('[','').replace(']','').strip().strip('\n')

                if score.isdigit() == False and score not in ['True','False'] and score != '':
                    score = extract_numbers(score)[0]
                elif score.isdigit():
                    score = float(score)
                elif score == 'True':
                    score = 1.0
                elif score == 'False':
                    score = 0.0
                res[name] = float(score)

        if name not in res:
            print('not found', name)
            res[name] = None
    
    return res

import re
def remove_text_in_brackets(text):
    return re.sub(r'\(.*\)', '', text)

def remove_text_in_stars(text):
    return re.sub(r"\*[^*]*\*", "", text)

def remove_emojis(data):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002500-\U00002BEF"
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642" 
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"
                           u"\u3030"
                           "]+", re.UNICODE)
    return emoji_pattern.sub(r' ', data)

def remove_noise(s):
    s = s.replace('’',"'").replace('‘',"'").replace('–', ' - ').replace('—',' - ').replace('“','"').replace('”','"').replace('…','.')
    s = re.sub(r'#\S+', '', s)
    s = remove_emojis(s)
    s = remove_text_in_brackets(s)
    s = remove_text_in_stars(s)
    s = ' '.join(s.split())
    s = s.replace(' # ','')
    s = s.strip()
    return s

import timeout_decorator
def gpt4_generate(prompt):
    return call_api([prompt], model_name = 'gpt-4o', temperature=0.0, max_tokens = 1024)
def chat_generate(prompts, temp = 0.5):
    return call_api(prompts, model_name = 'gpt-3.5-turbo', temperature=temp, max_tokens = 1024)

import requests
@timeout_decorator.timeout(50)
def together_generate(prompt, model = "meta-llama/Llama-3-70b-chat-hf", stop_tok = "<|eot_id|>",
                      temp = 0.8, system = None, max_tokens = 512):
    try:
        endpoint = 'https://api.together.xyz/v1/chat/completions'
        messages = [{"content": prompt, "role": "user"}]
        
        if system != None:
            messages.append({"content": system, "role": "system"})
        
        res = requests.post(endpoint, json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": [stop_tok],
            "messages": messages
        }, headers={
            "Authorization": TOGETHER_KEY,
        })
        return res.json()['choices'][0]['message']['content']
    except:
        print('api error!')
        return gpt4_generate(prompt)

def llama_generate(prompt, temp = 0.1, system = None, max_tokens = 512):
    return together_generate(prompt, model = "meta-llama/Llama-3-70b-chat-hf", stop_tok = "<|eot_id|>",
                             temp = temp, system = system, max_tokens = max_tokens)

@timeout_decorator.timeout(100)
def base_api(messages, model_name = 'gpt-3.5-turbo', max_tokens = 500, temperature = 1.0, n = 1):
    return client.chat.completions.create(model=model_name,
                                        temperature=temperature,
                                        messages=messages,
                                        max_tokens=max_tokens,
                                        n=n,
                                       )

def call_api(messages, model_name = 'gpt-3.5-turbo', 
             max_tokens = 500, temperature = 1.0, n = 1):
    
    if type(messages[0]) != dict:
        for i in range(0,len(messages)):
            if i % 2 == 0:
                messages[i] = {"role": "user", "content": messages[i]}
            else:
                messages[i] = {"role": "system", "content": messages[i]}
    
    result = base_api(messages, model_name, max_tokens, temperature, n = n)
    if n == 1:
        return result.choices[0].message.content
    else:
        return [line.message.content for line in result.choices]
    
    return None