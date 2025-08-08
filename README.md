# RDS
This is the official repository for "Root Defense Strategies: Ensuring Safety of LLM at the Decoding Level" (Accepted by ACL 2025).
## How to train the classifier
We train the classifier based on the work "On Prompt-Driven Safeguarding for Large Language Models" (https://github.com/chujiezheng/LLM-Safeguard/blob/main/code)
1. To generate the hidden state of the queries, run:
```
bash scripts/forward.sh
```
2. Replace the estimate.py with our released one, as we only use the hidden state of the original queries without any safety prompt.
3. To Train the classifier, run:
```
python estimate.py \
    --system_prompt_type ${system_prompt_type} \
    --config sampling --pretrained_model_path ${model}
```
   
## Add the classifier to the decoder layer of LLMs
During refering, RDS extends EAGLE_Head (https://github.com/SafeAILab/EAGLE) in resampling process to generate the hidden state of the candidate tokens. 
1. replace EAGLE-my/eagle/model/cnet.py with our cnet.py, in which we add the classifier in the decoder layer for resampling.
2. To generate the safe output, run:
```
python code.py 
```

