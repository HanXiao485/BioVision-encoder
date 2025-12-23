import torch
import os

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints'):
    """
    保存训练过程中的模型权重
    :param state: 包含 epoch, model_state, optimizer_state, acc 等的字典
    :param is_best: 是否为当前最优模型
    :param checkpoint_dir: 保存目录
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 1. 保存当前最新的 checkpoint (用于断点续训)
    last_path = os.path.join(checkpoint_dir, 'checkpoint_last.pth')
    torch.save(state, last_path)

    # 2. 如果是当前表现最好的模型，单独存一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth')
        # 我们通常只保存 state_dict (模型权重)，这样加载时更灵活
        torch.save(state['state_dict'], best_path)
        print(f"==> Saved Best Model (Acc: {state['acc']:.2f}%) to {best_path}")