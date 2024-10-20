import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from protein_dynamics.datasets import ProteinDynamicsDatasets
from protein_dynamics.proteindynamics import ProteinDynamic

def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def train(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    # 创建数据集和数据加载器
    train_dataset = ProteinDynamicsDatasets(args.train_csv)
    val_dataset = ProteinDynamicsDatasets(args.val_csv)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
    # for batch in train_loader:
    #     import pdb; pdb.set_trace()
    # 创建模型
    model = ProteinDynamic().to(device)
    model = DDP(model, device_ids=[rank])
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.4f}")
    
    # 保存模型
    if rank == 0:
        torch.save(model.module.state_dict(), args.model_path)

def inference(args):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinDynamic().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # 创建推理数据集和数据加载器
    test_dataset = ProteinDynamicsDatasets(args.test_csv)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 进行推理
    results = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'].to(device)
            outputs = model(inputs)
            results.append(outputs.cpu())
    
    # 处理结果
    # ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default='/home/bingxing2/gpuuser834/protein_dynamics/data/pdb_dataset.csv')
    parser.add_argument("--val_csv", type=str, default='/home/bingxing2/gpuuser834/protein_dynamics/data/pdb_dataset.csv')
    parser.add_argument("--test_csv", type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="protein_dynamic_model.pth")
    args = parser.parse_args()

    setup()
    train(args)
    cleanup()
    
    if args.test_csv:
        inference(args)