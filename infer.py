

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from model_class import BigramLanguageModel
torch.manual_seed(1337)
from bigram_dataset import MyDataset
device = "cuda:2" if torch.cuda.is_available() else "cpu"
from torch.utils.data import DataLoader
# device = "cpu"
# print(device)

batch_size = 64
block_size = 256
num_epochs = 5
eval_interval = 500
eval_itr = 200
embed_dim = 384
lr = 3e-4
max_new_tokens=2000

@torch.no_grad()
def infer(model, decode):
    model.eval()
    idx = torch.zeros((1, 1), dtype = torch.long)
    idx = idx.to(device)
    enc_output = model.generate(idx = idx, max_new_tokens=max_new_tokens)
    print("Output:", decode(enc_output[0].tolist()))
    model.train()
        
def train_test_split_data(data):
    n = int(0.9*(len(text)))
    return data[:n], data[n: ]

def preprocess_data(text, encode_text_fun):
    data = encode_text_fun(text)
    data = torch.tensor(data)
    return data
    
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, test_data):
    model.eval()
    losses = [0, 0]
    for i, data in enumerate([train_data, test_data]):
        for j in range(eval_itr):
            xb, yb = get_batch(data)
            logits, loss = model(xb, yb)
            losses[i] += loss.item()
        
    model.train()
    return losses[0]/eval_itr, losses[1]/eval_itr


if __name__ == "__main__":
    
    input_path = "/home/kadaru_teja/naveen/personal/study/gpt_from_scratch/source_code_actual/ng-video-lecture/input.txt"
    
    checkpoint_path = "/home/kadaru_teja/naveen/personal/study/gpt_from_scratch/models/expt4_ddp/checkpoint_epoch_0.pt"
    # os.makedirs(checkpoint_path, exist_ok=True)

    with open(input_path) as F:
        text = F.read()
        
    chars_set = sorted(set(text))
    stoi = {ch : i for i, ch in enumerate(chars_set)}
    itos = {i: ch for i, ch in enumerate(chars_set)}

    encode_text_fun = lambda s: [stoi[ch] for ch in s]
    decode_text_fun = lambda a: "".join([itos[x] for x in a])
    vocab_size = len(chars_set)
    model = BigramLanguageModel(vocab_size, embed_dim, block_size, device)
    model = model.to(device)
    
    ckpt_path = "/home/kadaru_teja/naveen/personal/study/gpt_from_scratch/models/expt4_ddp/checkpoint_epoch_4.pt"
    
    # model.module.load_state_dict(torch.load(ckpt_path, weights_only = False))
    
    
    
    checkpoint = torch.load(ckpt_path, weights_only=False)
    # Handle DDP state dict by removing 'module.' prefix
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    infer(model, decode_text_fun)