from utils import add_personAB

START_INST = '[INST] '
END_INST = ' [/INST] '

def dialog_listing(context, N):
    prompt = "Given this conversation:\n\n"
    prompt += add_personAB(context) + '\n\n'
    prompt += "Imagine you are person B and act as if you were a real individual. Think about all the possibilities in which person B might respond next and then provide a list of " + str(N) + " different diverse responses. Keep each response less than 25 words and semantically different from one another. Also make sure that each response is coherent and relevant with the conversation history." + END_INST + "Person B's response 1:"
    return prompt

def dialog_prompt(context, option = None, tuning = False):
    prompt  = "Given this conversation:\n\n"
    prompt += add_personAB(context) + "\n\n"
    
    if tuning is False:
        if type(option) != int:
            prompt += "Imagine you are person B and act as if you were a real individual. Please write the next response for person B. Keep the response short with no more than 25 words." + END_INST + "Person B:"
        else:
            prompt += "Imagine you are person B and act as if you were a real individual. Think about all the possibilities in which person B might respond next and then provide the response that corresponds to possibility number #" + str(option) + ". Keep the response short with no more than 25 words." + END_INST + "Person B:"
    else:
        if type(option) != int:
            prompt += "Imagine you are person B and act as if you were a real individual. Please write the next response for person B. " +  END_INST.strip(' ')
        else:
            prompt += "Imagine you are person B and act as if you were a real individual. Think about all the possibilities in which person B might respond next and then provide the response that corresponds to possibility number #" + str(option) + ". " + END_INST.strip(' ')
    prompt  = START_INST + prompt
    return prompt
