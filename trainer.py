import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from utils.metrics import calculate_accuracy

import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.transformer import TransformerClassifier
from models.vit import VisionTransformer

import pytorch_lightning as pl
from datasets.cifar import CIFAR100Manager
from datasets.imagenet import ImageNetManager


class Classifier(pl.LightningModule):
    def __init__(self, 
                 model, 
                 lr=1e-3,
                 weight_decay=1e-4,
                 scheduler_type="step",     # step | cosine | onecycle | plateau
                 step_size=5,
                 gamma=0.1,
                 min_lr=1e-6,
                 max_lr=1e-3,):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["model"])


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc * 100.0, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.step_size,
                gamma=self.gamma,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        elif self.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        elif self.scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        elif self.scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.gamma,
                patience=3,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val/acc",
                },
            }

        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dm = CIFAR100Manager(batch_size=self.batch_size)
        self.train_loader = self.dm.get_loader(train=True)
        self.val_loader = self.dm.get_loader(train=False)

        # input_dim / num_classes
        sample_batch, _ = next(iter(self.train_loader))
        c, h, w = sample_batch.shape[1:]
        self.input_dim = c * h * w
        self.num_classes = len(set(self.train_loader.dataset.targets))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

class IMAGENETDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dm = ImageNetManager(batch_size=self.batch_size)
        self.train_loader = self.dm.get_loader(train=True)
        self.val_loader = self.dm.get_loader(train=False)

        # input_dim / num_classes
        sample_batch, _ = next(iter(self.train_loader))
        c, h, w = sample_batch.shape[1:]
        self.input_dim = c * h * w
        self.num_classes = len(set(self.train_loader.dataset.targets))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def export_model(ckpt_path: str, export_dir: str):
    """
    recover the LitModule from checkpoint,
    extract the underlying nn.Module,
    and save the full model for deployment.
    Args:
        ckpt_path (str): path to the checkpoint file.
        export_dir (str): directory to save the exported model.
    Returns:
        None
    """

    # 1. recover the LitModule
    lit_model = Classifier.load_from_checkpoint(
        ckpt_path,
        model=VisionTransformer(
            img_size=32,
            patch_size=2,
            in_channels=3,
            num_classes=100,
            embed_dim=512,
            num_layers=6,
            num_heads=8,
            mlp_dim=1024,
        ),
        map_location="cpu", 
    )

    lit_model.eval()

    # 2. extract the underlying nn.Module
    deploy_model = lit_model.model
    deploy_model.eval()
    deploy_model.cpu()

    # save the full model
    full_model_path = os.path.join(export_dir, "vit_full_model.pt")
    torch.save(deploy_model, full_model_path)
    print(f"Saved full PyTorch model to {full_model_path}")

    # # TorchScript export (optional)
    # example_input = torch.randn(1, 3, 32, 32)

    # scripted_model = torch.jit.trace(deploy_model, example_input)
    # scripted_model = torch.jit.freeze(scripted_model)
    # ts_path = os.path.join(export_dir, "vit_torchscript.pt")
    # scripted_model.save(ts_path)

    # print(f"Saved TorchScript model to {ts_path}")


def main(args):

    if args.gpus is not None:
        # 例：--gpus 0,1,3
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        gpu_list = args.gpus.split(",")
        devices = len(gpu_list)
        accelerator = "gpu"
        strategy = "ddp" if devices > 1 else "auto"
    else:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = "auto"
        strategy = "auto"

    # 1. DataModule
    # dm = CIFARDataModule(batch_size=args.batch_size)
    dm = IMAGENETDataModule(batch_size=args.batch_size)
    dm.setup()

    # 2. Model
    # backbone = TransformerClassifier(
    #     input_dim=dm.input_dim,
    #     hidden_dim=512,
    #     num_tokens=256,
    #     num_heads=32,
    #     num_layers=8,
    #     num_classes=dm.num_classes,
    # )
    backbone = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1
    )


    model = Classifier(
        backbone,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
    )

    # 3. Logger
    logger = None
    if args.use_tb:
        logger = TensorBoardLogger(
            save_dir="runs",
            name=args.exp_name,
        )

    # 4. Checkpoint
    ckpt_callback = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename="epoch{epoch}-acc{val/acc:.2f}",
        auto_insert_metric_name=False,
    )

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy="ddp" if devices != 1 else "auto",
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt_callback],
        log_every_n_steps=10,
    )

    # 6. Train
    trainer.fit(model, datamodule=dm)

    # 7. Export the best model
    if trainer.is_global_zero:
        best_ckpt_path = ckpt_callback.best_model_path
        print(f"Best checkpoint: {best_ckpt_path}")

        export_dir = "exported_models"
        os.makedirs(export_dir, exist_ok=True)

        export_model(
            ckpt_path=best_ckpt_path,
            export_dir=export_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tb", default=True, action="store_true")
    parser.add_argument("--exp_name", type=str, default="ViT-B16-imagenet1K")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay",type=float,default=0.3,help="Weight decay for AdamW")
    parser.add_argument("--scheduler", type=str, default="cosine",choices=["step", "cosine", "onecycle", "plateau"])
    parser.add_argument("--precision",type=str,default="16-mixed",choices=["32", "16", "bf16", "16-mixed", "bf16-mixed"],help="Training precision mode")
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--gpus", type=str, default=None)

    args = parser.parse_args()
    main(args)
