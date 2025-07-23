import os
import json
import csv
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
import torch
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from utils import patch_open, logging_cuda_memory_usage, get_following_indices
from safetensors import safe_open
import gc
import random
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from utils import PCA_DIM
import torch.nn as nn
from sklearn.metrics import roc_auc_score,roc_curve

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def calculate_boundary(xlim, ylim, weight, bias):
    if np.abs(weight[0]) > np.abs(weight[1]):
        xlim_by_ylim_0 = (-bias - weight[1] * ylim[0]) / weight[0]
        xlim_by_ylim_1 = (-bias - weight[1] * ylim[1]) / weight[0]
        return [(xlim_by_ylim_0, ylim[0]), (xlim_by_ylim_1, ylim[1])]
    else:
        ylim_by_xlim_0 = (-bias - weight[0] * xlim[0]) / weight[1]
        ylim_by_xlim_1 = (-bias - weight[0] * xlim[1]) / weight[1]
        return [(xlim[0], ylim_by_xlim_0), (xlim[1], ylim_by_xlim_1)]


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_paths", type=str, nargs='+', required=True)
    parser.add_argument("--system_prompt_type", type=str, choices=['all'], required=True)
    parser.add_argument("--config", type=str, choices=["greedy", "sampling"])
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # prepare data
    fname = f'{args.system_prompt_type}_harmfulness_boundary'
    fname += "_custom"
    dataset = 'custom'
    with open(f"./data/custom.txt") as f:
        lines = f.readlines()
    with open(f"./data_harmless/custom.txt") as f:
        lines_harmless = f.readlines()
    os.makedirs(args.output_path, exist_ok=True)

    #colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        'harmless': 'tab:blue',
        'harmful': 'tab:red',
        'harmless + 1': 'tab:cyan',
        'harmful + 1': 'black',
        'harmless + 2': 'tab:olive',
        'harmful + 2': 'tab:purple',
        'harmless + 3': 'tab:brown',
        'harmful + 3': 'tab:orange',
        'harmless + 4': 'tab:orange',
        'harmful + 4': 'tab:olive',
    }

    all_queries = [e.strip() for e in lines if e.strip()]
    n_queries = len(all_queries)

    all_queries_harmless = [e.strip() for e in lines_harmless if e.strip()]
    n_queries_harmless = len(all_queries_harmless)

    ncols = 1
    if len(args.pretrained_model_paths) % ncols != 0:
        raise ValueError(f"len(args.pretrained_model_paths) % ncols != 0")
    nrows = len(args.pretrained_model_paths) // ncols
    fig = plt.figure(figsize=(4.5 * ncols, 4.5 * nrows))

    

    for mdx, pretrained_model_path in enumerate(args.pretrained_model_paths):
        logging_cuda_memory_usage()
        torch.cuda.empty_cache()
        gc.collect()

        logging.info(pretrained_model_path)

        # prepare model
        model_name = pretrained_model_path.split('/')[-1]
        config = AutoConfig.from_pretrained(pretrained_model_path)
        num_layers = config.num_hidden_layers
        harmfulness_model = nn.Linear(PCA_DIM, 1)
        with safe_open(f'estimations/{model_name}_all/harmfulness.safetensors', framework='pt') as f:
            weight = f.get_tensor('weight').mean(dim=0)
            bias = f.get_tensor('bias').mean(dim=0)
        harmfulness_model.load_state_dict({'weight': weight, 'bias': bias})
        harmfulness_model.float().to('cuda')
        for param in harmfulness_model.parameters():
            param.requires_grad = False
        


        # w/o
        logging.info(f"Running w/o")
        hidden_states = safe_open(f'hidden_states/{model_name}_{dataset}.safetensors',
                                framework='pt', device=0)
        hidden_states_with_default = safe_open(f'hidden_states/{model_name}_with_default_{dataset}.safetensors',
                                            framework='pt', device=0)
        hidden_states_with_short = safe_open(f'hidden_states/{model_name}_with_short_{dataset}.safetensors',
                                            framework='pt', device=0)
        hidden_states_with_mistral = safe_open(f'hidden_states/{model_name}_with_mistral_{dataset}.safetensors',
                                            framework='pt', device=0)
        all_hidden_states = []
        all_hidden_states_with_default = []
        all_hidden_states_with_short = []
        all_hidden_states_with_mistral = []
        for idx, query in enumerate(all_queries):
            tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
            tmp_hidden_states_with_default = hidden_states_with_default.get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
            tmp_hidden_states_with_short = hidden_states_with_short.get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
            tmp_hidden_states_with_mistral = hidden_states_with_mistral.get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
            all_hidden_states.append(tmp_hidden_states)
            all_hidden_states_with_default.append(tmp_hidden_states_with_default)
            all_hidden_states_with_short.append(tmp_hidden_states_with_short)
            all_hidden_states_with_mistral.append(tmp_hidden_states_with_mistral)

        # harmless
        logging.info(f"Running harmless")
        hidden_states = safe_open(f'hidden_states_harmless/{model_name}_{dataset}.safetensors',
                                    framework='pt', device=0)
        hidden_states_with_default = safe_open(f'hidden_states_harmless/{model_name}_with_default_{dataset}.safetensors',
                                            framework='pt', device=0)
        hidden_states_with_short = safe_open(f'hidden_states_harmless/{model_name}_with_short_{dataset}.safetensors',
                                            framework='pt', device=0)
        hidden_states_with_mistral = safe_open(f'hidden_states_harmless/{model_name}_with_mistral_{dataset}.safetensors',
                                            framework='pt', device=0)
        all_hidden_states_harmless = []
        all_hidden_states_with_default_harmless = []
        all_hidden_states_with_short_harmless = []
        all_hidden_states_with_mistral_harmless = []
        for idx, query_harmless in enumerate(all_queries_harmless):
            tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
            tmp_hidden_states_with_default = hidden_states_with_default.get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
            tmp_hidden_states_with_short = hidden_states_with_short.get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
            tmp_hidden_states_with_mistral = hidden_states_with_mistral.get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
            all_hidden_states_harmless.append(tmp_hidden_states)
            all_hidden_states_with_default_harmless.append(tmp_hidden_states_with_default)
            all_hidden_states_with_short_harmless.append(tmp_hidden_states_with_short)
            all_hidden_states_with_mistral_harmless.append(tmp_hidden_states_with_mistral)


        all_hidden_states = torch.stack(all_hidden_states)
        all_hidden_states_with_default = torch.stack(all_hidden_states_with_default)
        all_hidden_states_with_short = torch.stack(all_hidden_states_with_short)
        all_hidden_states_with_mistral = torch.stack(all_hidden_states_with_mistral)
        all_hidden_states_harmless = torch.stack(all_hidden_states_harmless)
        all_hidden_states_with_default_harmless = torch.stack(all_hidden_states_with_default_harmless)
        all_hidden_states_with_short_harmless = torch.stack(all_hidden_states_with_short_harmless)
        all_hidden_states_with_mistral_harmless = torch.stack(all_hidden_states_with_mistral_harmless)

        indices, other_indices = get_following_indices(
            model_name, config=args.config, use_harmless=False)
        indices_harmless, other_indices_harmless = get_following_indices(model_name, config=args.config, use_harmless=True)

        
        all_list = [all_hidden_states_harmless]
        names = globals()
        
        for i in range(1,10):#,2,3,4,5,10,20,30,40,50,60,70,80,90,100]:
            names[f'hidden_states_with_{i}_harmless'] = safe_open(f'pre_hidden_states_harmless/{model_name}_{dataset}_{i}.safetensors',
                                    framework='pt', device=0)
            names[f'all_hidden_states_with_{i}_harmless'] = list()

            for idx, query in enumerate(all_queries_harmless):
                names[f'tmp_hidden_states_with_{i}_harmless'] = names[f'hidden_states_with_{i}_harmless'].get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
                names[f'all_hidden_states_with_{i}_harmless'].append(names[f'tmp_hidden_states_with_{i}_harmless'])
            names[f'all_hidden_states_with_{i}_harmless'] = torch.stack(names[f'all_hidden_states_with_{i}_harmless'])
            all_list.append(names[f'all_hidden_states_with_{i}_harmless'])
        all_list.append(all_hidden_states)
        


        for i in range(1,10):#,2,3,4,5,10,20,30,40,50,60,70,80,90,100]:
            names[f'hidden_states_with_{i}'] = safe_open(f'pre_hidden_states/{model_name}_{dataset}_{i}.safetensors',
                                    framework='pt', device=0)
            names[f'all_hidden_states_with_{i}'] = list()

            for idx, query in enumerate(all_queries):
                names[f'tmp_hidden_states_with_{i}'] = names[f'hidden_states_with_{i}'].get_tensor(f'sample.{idx}_layer.{num_layers - 1}')[-1]
                names[f'all_hidden_states_with_{i}'].append(names[f'tmp_hidden_states_with_{i}'])
            names[f'all_hidden_states_with_{i}'] = torch.stack(names[f'all_hidden_states_with_{i}'])
            all_list.append(names[f'all_hidden_states_with_{i}'])
        #
        '''
        hidden_states = torch.cat([
        all_hidden_states_harmless,
        all_hidden_states_with_default_harmless,
        all_hidden_states_with_short_harmless,
        all_hidden_states_with_mistral_harmless,
        all_hidden_states,
        all_hidden_states_with_default,
        all_hidden_states_with_short,
        all_hidden_states_with_mistral,
    ], dim=0).float()
        '''
        hidden_states = torch.cat([all_hidden_states, all_hidden_states_harmless,], dim=0).float()
        #hidden_states = torch.cat(all_list, dim=0).float()

        
        pca = PCA(PCA_DIM, random_state=42)
        pca.fit(hidden_states.cpu().numpy())
        mean = torch.tensor(pca.mean_, device='cuda', dtype=torch.float)
        V = torch.tensor(pca.components_.T, device='cuda', dtype=torch.float)
        logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}, sum: {np.sum(pca.explained_variance_ratio_)}")



        
        for j in range(1,10):

            # harmless
            '''
            points = torch.matmul(all_hidden_states_harmless - mean, V)[:, 0:].cpu().numpy()
            ax.scatter(points[:, 0], points[:, 1],
                        marker='o', alpha=0.39,
                        color=colors['harmless'], label='harmless')

            points = torch.matmul(all_hidden_states - mean, V)[:, 0:].cpu().numpy()
            
            ax.scatter(points[:, 0], points[:, 1],
                        marker='x', alpha=0.41,
                        color=colors['harmful'], label='harmful')
            '''
            # indices 回答 other拒答
            if model_name in ['openchat-3.5','vicuna-7B-1.3','vicuna-13B-1.3','Llama-2-7b-chat-hf','Mistral-7B-Instruct-v0.1']:
                t = j - 1
            else:
                t = j
            '''
            ax.scatter(points[other_indices, 0], points[other_indices, 1],
                        marker='x', alpha=0.42,
                        color='tab:olive', label='refusal+'+str(t))
            
            ax.scatter(points[indices, 0], points[indices, 1],
                        marker='x', alpha=0.41,
                        color='black', label='compliance+'+str(t))
             
            points = torch.matmul(names[f'all_hidden_states_with_{j}_harmless'] - mean, V)[:, 0:].cpu().numpy()
            
        
            ax.scatter(points[other_indices_harmless, 0], points[other_indices_harmless, 1],
                        marker='o', alpha=0.42,
                        color= 'gray', label='refusal+'+str(t))
            
            ax.scatter(points[indices_harmless, 0], points[indices_harmless, 1],
                        marker='o', alpha=0.41,
                        color='green', label='compliance+'+str(t))
            '''
            '''
            points = torch.matmul(names[f'all_hidden_states_with_{j}_harmless'] - mean, V)[:, 0:].cpu().numpy()
            ax.scatter(points[:, 0], points[:, 1],
                        marker='o', alpha=0.42,
                        color= 'tab:olive', label='harmless+'+str(t))
            '''
            points = torch.matmul(all_hidden_states - mean, V)
            data = harmfulness_model(points[5]).squeeze(-1).sigmoid().cpu().detach().numpy()
            print(data)
            points = torch.matmul(names[f'all_hidden_states_with_{j}'] - mean, V)
            data1 = harmfulness_model(points[other_indices]).sigmoid().squeeze(-1).cpu().detach().numpy()
            data2 = harmfulness_model(points[indices]).sigmoid().squeeze(-1).cpu().detach().numpy()

            plt.figure(figsize=(10, 6))
            plt.scatter(-1, data, color='black', label=f'query')
            plt.scatter(range(len(data2)), data2, color='salmon', label=f'Distribution harmful+{j}')
            plt.scatter(range(len(data2), len(data2)+len(data1)), data1, color='skyblue', label=f'Distribution harmless+{j}')
            plt.legend()
            plt.title('Comparison of Two Distributions')
            plt.xlabel('Samples')
            plt.ylabel('Values')
            plt.savefig(f"{args.output_path}/{model_name}_{j}_distribution.jpg")
            plt.show()

if __name__ == "__main__":
    main()
