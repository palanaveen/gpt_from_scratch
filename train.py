import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from naveen.personal.study.gpt_from_scratch.model import BigramLanguageModel
import torch.multiprocessing as mp
torch.manual_seed(1337)
from naveen.personal.study.gpt_from_scratch.dataset import MyDataset
# device = "cuda:2" if torch.cuda.is_available() else "cpu"
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
# device = "cpu"
# print(device)

batch_size = 256
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
    idx = idx.to("cuda:3")
    enc_output = model.generate(idx = idx, max_new_tokens=max_new_tokens)
    print("Output:", decode(enc_output[0].tolist()))
    model.train()
        
def train_test_split_data(data):
    n = int(0.9*(len(data)))
    return data[:n], data[n: ]

def preprocess_data(text, encode_text_fun, vocab):
    data = encode_text_fun(text, vocab)
    data = torch.tensor(data)
    return data
    
# def get_batch(data):
#     ix = torch.randint(len(data) - block_size, (batch_size, ))
#     x = torch.stack([data[i:i + block_size] for i in ix])
#     y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
#     x = x.to(device)
#     y = y.to(device)
#     return x, y

# @torch.no_grad()
# def estimate_loss(model, train_data, test_data):
#     model.eval()
#     losses = [0, 0]
#     for i, data in enumerate([train_data, test_data]):
#         for j in range(eval_itr):
#             xb, yb = get_batch(data)
#             logits, loss = model(xb, yb)
#             losses[i] += loss.item()
        
#     model.train()
#     return losses[0]/eval_itr, losses[1]/eval_itr
def encode_text_fun(s, vocab):
    stoi = {ch : i for i, ch in enumerate(vocab)}
    return [stoi[ch] for ch in s]

def decode_text_fun(a, vocab):
    itos = {i: ch for i, ch in enumerate(vocab)}
    return  "".join([itos[x] for x in a])
    
    
def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "127.0.0.1"
   os.environ["MASTER_PORT"] = "34336"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size) 
   
def main(rank, world_size, text, vocab, checkpoint_path):
    ddp_setup(rank, world_size)
    vocab_size = len(vocab)
    device = f"cuda:{rank}"
    model = BigramLanguageModel(vocab_size, embed_dim, block_size, device)

    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    train_text, val_text = train_test_split_data(text)
    train_data = preprocess_data(train_text, encode_text_fun, vocab)
    dataset = MyDataset(train_data, block_size)
    
    train_data = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers=8, sampler=DistributedSampler(dataset))
    
    # val_data = preprocess_data(val_text, encode_text_fun, vocab)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Training loop
    # print("Check 3")
    
    for epoch in range(num_epochs):
        # print("Check")
        train_data.sampler.set_epoch(epoch)
        if rank == 0:
            print(f"Epoch {epoch}")
        for i, (xb, yb) in enumerate(train_data):
            # xb, yb = get_batch(train_data) #shape of xb and yb: batch_size x block_size
            xb = xb.to(device)
            yb = yb.to(device)
            logits, loss = model(idx = xb, targets = yb)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if rank == 0:
                print(i)
            
        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch}.pt"))
            # break
        
        # if(epoch % eval_interval == 0):
        #     train_loss, val_loss = estimate_loss(model, train_data, val_data)
        #     print(f"Step: {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
            
        # print("Final Loss: ", loss)
    destroy_process_group()

    


if __name__ == "__main__":
    start_time = time.time()
    input_path = "/home/kadaru_teja/naveen/personal/study/gpt_from_scratch/source_code_actual/ng-video-lecture/input.txt"

    checkpoint_path = "/home/kadaru_teja/naveen/personal/study/gpt_from_scratch/models/expt4_ddp"
    os.makedirs(checkpoint_path, exist_ok=True)

    with open(input_path) as F:
        text = F.read()
        
    vocab = sorted(set(text))
    
    # main(model, text, encode_text_fun, decode_text_fun, vocab_size, checkpoint_path)
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, text, vocab, checkpoint_path), nprocs=world_size)
    end_time = time.time()
    print("Time:", int(end_time - start_time))