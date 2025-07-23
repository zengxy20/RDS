import os
import json
import csv
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
import warnings
from utils import patch_open, logging_cuda_memory_usage, get_following_indices
from utils import PCA_DIM
from safetensors import safe_open
import gc
import random
from matplotlib import pyplot as plt
from safetensors.torch import save_file
from sklearn.decomposition import PCA
from copy import deepcopy
from utils import gram_schmidt


logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")


def kmeans_smoothing(coordinates, values, k=5):
    max_distance = torch.max(torch.norm(coordinates.unsqueeze(0) - coordinates.unsqueeze(1), dim=-1))
    smoothed_values = []
    for i in range(len(coordinates)):
        if k == 1:
            smoothed_value = values[i]
        else:
            current_coord = coordinates[i]
            distances = torch.norm(coordinates - current_coord, dim=1)
            _, indices = torch.topk(distances, k=k, largest=False)
            weights = torch.exp(-distances[indices] / max_distance / 0.2)
            smoothed_value = torch.sum(values[indices] * weights) / torch.sum(weights)
        smoothed_values.append(smoothed_value.item())
    smoothed_values = np.array(smoothed_values)
    return smoothed_values


MAX_EPOCHES = 10000


def main():
    patch_open()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--config", type=str, choices=["greedy", "sampling"], required=True)
    parser.add_argument("--output_path", type=str, default='./estimations_xstest_middle1')
    parser.add_argument("--system_prompt_type", type=str, choices=['all'], required=True)
    parser.add_argument("--n_splits", type=int, default=10)
    args = parser.parse_args()

    # logging args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    # prepare data
    dataset = 'xstest'
    with open(f"./data/{dataset}.txt") as f:
        lines = f.readlines()
    with open(f"./data_harmless/{dataset}.txt") as f:
        lines_harmless = f.readlines()

    all_queries = [e.strip() for e in lines if e.strip()]
    n_queries = len(all_queries)

    all_queries_harmless = [e.strip() for e in lines_harmless if e.strip()]
    n_queries_harmless = len(all_queries_harmless)

    assert n_queries == n_queries_harmless, f"{n_queries} != {n_queries_harmless}"
    assert n_queries % args.n_splits == 0, f"{n_queries} % {args.n_splits} != 0"

    logging_cuda_memory_usage()
    torch.cuda.empty_cache()
    gc.collect()

    logging.info(args.pretrained_model_path)

    # prepare model
    model_name = args.pretrained_model_path.split('/')[-1]
    config = AutoConfig.from_pretrained(args.pretrained_model_path)
    num_layers = config.num_hidden_layers
    mid_layers = num_layers//2 + 1

    os.makedirs(f'{args.output_path}/{model_name}_{args.system_prompt_type}', exist_ok=True)

    hidden_states = safe_open(f'hidden_states/{model_name}_{dataset}.safetensors',
                                  framework='pt', device=0)
    all_hidden_states = []
    for idx, query in enumerate(all_queries):
        tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{mid_layers}')[-1]
        all_hidden_states.append(tmp_hidden_states)

    hidden_states = safe_open(f'hidden_states_harmless/{model_name}_{dataset}.safetensors',
                                framework='pt', device=0)
    all_hidden_states_harmless = []
    for idx, query_harmless in enumerate(all_queries_harmless):
        tmp_hidden_states = hidden_states.get_tensor(f'sample.{idx}_layer.{mid_layers}')[-1]
        all_hidden_states_harmless.append(tmp_hidden_states)

    all_hidden_states = torch.stack(all_hidden_states)
    all_hidden_states_harmless = torch.stack(all_hidden_states_harmless)
    all_list = [all_hidden_states, all_hidden_states_harmless]
    
    hidden_states = torch.cat(all_list, dim=0).float()

    pca = PCA(PCA_DIM, random_state=42)
    pca.fit(hidden_states.cpu().numpy())
    logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}, sum: {np.sum(pca.explained_variance_ratio_)}")

    mean = torch.tensor(pca.mean_, dtype=torch.float, device='cuda')
    V = torch.tensor(pca.components_.T, dtype=torch.float, device='cuda')
    n = V.size(0)
    basis = [V[:, i] for i in range(V.size(1))]
    set_seed(42)
    all_vectors = torch.randn((n*2, n), device='cuda', dtype=torch.double)
    orthogonal_basis = gram_schmidt(all_vectors, basis, n)
    save_file({'mean': mean, 'V': orthogonal_basis}, f'{args.output_path}/{model_name}_{args.system_prompt_type}/transform.safetensors')


    def train_model(model, train_X, train_Y, test_X=None, test_Y=None):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        min_epoch = 0
        min_test_bce_loss = 1e5
        min_test_mse_loss_by_bce_loss = 1e5
        epoch = 0
        test_loss_drop_times = 0
        best_model_copy = deepcopy(model)
        while True:
            model.train()
            optimizer.zero_grad()
            train_logits = model(train_X).squeeze(-1)
            bce_loss = F.binary_cross_entropy_with_logits(train_logits, train_Y)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            bce_loss.backward()
            optimizer.step()
            epoch += 1

            if test_X is not None:
                with torch.no_grad():
                    model.eval()
                    test_logits = model(test_X).squeeze(-1)
                    test_bce_loss = F.binary_cross_entropy_with_logits(test_logits, test_Y).detach()
                    test_mse_loss = F.mse_loss(test_logits.sigmoid(), test_Y).detach()

            else:
                test_bce_loss = bce_loss.detach()
                test_mse_loss = F.mse_loss(train_logits.sigmoid(), train_Y).detach()

            if min_test_bce_loss is None or test_bce_loss < min_test_bce_loss - 1e-4:
                min_test_bce_loss = test_bce_loss
                min_test_mse_loss_by_bce_loss = test_mse_loss
                best_model_copy = deepcopy(model)
                test_loss_drop_times = 0
                min_epoch = epoch
            else:
                test_loss_drop_times += 1

            if epoch == MAX_EPOCHES or test_loss_drop_times == 3:
                logging.info(f"Final epoch {min_epoch}: bce_loss {min_test_bce_loss.item()}, mse_loss {min_test_mse_loss_by_bce_loss.item()}")
                break
        return best_model_copy


    total_indices = list(range(2*n_queries))
    set_seed(42)
    random.shuffle(total_indices)

    logging.info(f"Training harmfulness model")
    harmfulness_linear_weights = []
    harmfulness_linear_biases = []
    train_X = torch.cat(all_list, dim=0).float()
    train_X = torch.matmul(train_X - mean, V)
    train_Y = torch.tensor([0 for _ in range(1*n_queries)] + [1 for _ in range(1*n_queries)], device=train_X.device, dtype=torch.float)

    
    test_X, test_Y = None, None
    model_harmfulness = nn.Linear(4, 1).to('cuda')
    model_harmfulness = train_model(model_harmfulness, train_X, train_Y, test_X, test_Y)
    harmfulness_linear_weights.append(model_harmfulness.weight.detach())
    harmfulness_linear_biases.append(model_harmfulness.bias.detach())

    harmfulness_linear_weight = torch.stack(harmfulness_linear_weights).cpu()
    harmfulness_linear_bias = torch.stack(harmfulness_linear_biases).cpu()

    save_file({'weight': harmfulness_linear_weight, 'bias': harmfulness_linear_bias},
              f'{args.output_path}/{model_name}_{args.system_prompt_type}/harmfulness.safetensors')

    logging.info(f"Training finished")

    

if __name__ == "__main__":
    main()
