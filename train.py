import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

from datasets.cifar import CIFAR10Manager,CIFAR100Manager
from models.simple_mlp import SimpleMLP
from models.attention import AttentionNet
from models.transformer import TransformerClassifier
from utils.metrics import calculate_accuracy
from utils.checkpoint import save_checkpoint

def main(args):

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"==> Using GPUs: {args.gpus}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"==> Detected {num_gpus} GPU(s)")

    writer = None
    if args.use_tb:
        log_dir = f'runs/{args.exp_name}'
        writer = SummaryWriter(log_dir=log_dir)
        print(f"==> TensorBoard logging enabled at: {log_dir}")
    else:
        print("==> TensorBoard logging disabled.")
    
    # 1. load datasets
    dm = CIFAR100Manager(batch_size=args.batch_size)
    train_loader = dm.get_loader(train=True)
    test_loader = dm.get_loader(train=False)

    sample_batch, _ = next(iter(train_loader))
    input_shape = sample_batch.shape[1:]  # [channel, height, width]
    input_dim = input_shape[0] * input_shape[1] * input_shape[2] 
    num_classes = len(set(train_loader.dataset.targets))

    # 2. init models
    # model = AttentionNet(input_dim=input_dim, 
    #                      hidden_dim=args.hidden, 
    #                      num_classes=num_classes, 
    #                      num_heads=8).to(device)
    model = TransformerClassifier(
                            input_dim=input_dim,   
                            hidden_dim=1024,
                            num_tokens=128,  
                            num_heads=16,
                            num_layers=8,
                            num_classes=num_classes
                        ).to(device)

    if num_gpus > 1:
        print("==> Using DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)

    best_acc = 0.0
    
    # 3. define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 4. train
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # --- 3. batch level Loss ---
            if writer is not None and i % 100 == 99:
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Train/Loss', running_loss / 100, global_step)
                running_loss = 0.0

        acc = calculate_accuracy(model, test_loader, device)
        lr = optimizer.param_groups[0]['lr']

        if writer is not None:
            writer.add_scalar('Test/Accuracy', acc, epoch)
            writer.add_scalar('Train/Learning_Rate', lr, epoch)

        scheduler.step()

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': state_dict, # only weight
            'acc': acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        print(f"Epoch [{epoch+1}/{args.epochs}] Acc: {acc:.2f}%")

    if writer is not None:
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR10 Training")
    parser.add_argument('--use_tb', action='store_true', default=False, help='use_tensorboard')
    parser.add_argument('--exp_name', type=str, default='transformersGPU', help='exp_name')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--gpus', type=str, default=None, help='Comma separated GPU ids, e.g. "0,1,2"')
    args = parser.parse_args()
    
    main(args)