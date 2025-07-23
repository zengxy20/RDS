from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig

import pandas as pd
from tqdm import *
import json
from safetensors import safe_open
from sklearn.decomposition import PCA
import torch.nn as nn
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


model_name = 'LLM name'
base_model_path = 'your path of LLMs'
EAGLE_model_path = 'your path of pretrained EAGLE Head'
chat_template = open('your path of chat_templates').read()

config = AutoConfig.from_pretrained(base_model_path)
num_layers = config.num_hidden_layers
chat_template = chat_template.replace('    ', '').replace('\n', '')

harmfulness_model = nn.Linear(4, 1).to('cuda')
with safe_open(f'your path of classifier parameter (harmfulness.safetensors)', framework='pt') as f:
    weight = f.get_tensor('weight').mean(dim=0)
    bias = f.get_tensor('bias').mean(dim=0)
harmfulness_model.load_state_dict({'weight': weight, 'bias': bias})
harmfulness_model.float().to('cuda')
for param in harmfulness_model.parameters():
    param.requires_grad = False

hidden_states = safe_open(f'your path of harmful queries hidden state',
                            framework='pt', device='cuda')
all_hidden_states = []
midle_layer = num_layers//2
for idx in range(100):
    tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
    all_hidden_states.append(tmp_hidden_states)
hidden_states = safe_open(f'your path of harmless queries hidden state',
                            framework='pt', device='cuda')
all_hidden_states_harmless = []
for idx in range(100):
    tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers-1}')[-1]
    all_hidden_states_harmless.append(tmp_hidden_states)
all_hidden_states = torch.stack(all_hidden_states)
all_hidden_states_harmless = torch.stack(all_hidden_states_harmless)
hidden_states = torch.cat([all_hidden_states,all_hidden_states_harmless,], dim=0).float()

pca = PCA(4, random_state=42)
pca.fit(hidden_states.cpu().numpy())
mean = torch.tensor(pca.mean_, device='cuda', dtype=torch.float)
V = torch.tensor(pca.components_.T, device='cuda', dtype=torch.float)

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()
model.tokenizer.chat_template = chat_template

file='your path of the test data'
prompts = []
inputs = []
outputs = []
with open(file,'r') as f:
    lines = f.readlines()
all_queries = [l.strip() for l in lines][:100]
for i,your_message in enumerate(tqdm(all_queries)):
    prompt = prepend_sys_prompt(your_message)
    model.tokenizer.chat_template = chat_template
    
    prompt = model.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    
    input_ids=model.tokenizer(prompt).input_ids
    input_ids = torch.as_tensor(input_ids).cuda().unsqueeze(0)
    for num in range(5):#sampling 5 responses for each query
        output_ids=model.eagenerate(input_ids,mean, V, harmfulness_model,temperature=1,max_new_tokens=512)
        output=model.tokenizer.decode(output_ids[0][input_ids.size(1):])
        inputs.extend([prompt])
        prompts.extend([your_message])
        outputs.extend([output])

results = pd.DataFrame()
results["prompt"] = prompts
results["input"] = inputs
results["output"] = outputs
os.makedirs(f"your path of the output", exist_ok=True)
results.to_csv(f"your path of the output",mode='w')

