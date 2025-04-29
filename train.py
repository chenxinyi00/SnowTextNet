import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.plugins import DDPPlugin as DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import Text
from CL1 import L1_Charbonnier_loss, PSNRLoss
from perceptual import PerceptualLoss2
from metrics import PSNR, SSIM
from DGSNet import SnowFormer
from argparse import Namespace
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
import os

# Set global seed
seed = 42
seed_everything(seed)

# TensorBoard Logger
from pytorch_lightning.loggers import TensorBoardLogger

wandb_logger = TensorBoardLogger(r'/data1/cxy/data/DSR_ACMMM/tb_logs', name='SnowFormer')

class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()
        self.save_hyperparameters()  # 保存超参数

        self.params = hparams

        # Train/Test datasets
        self.train_datasets = self.params.train_datasets
        self.train_batchsize = 1  # 每次训练 2 张图片
        self.test_datasets = self.params.test_datasets
        self.test_batchsize = self.params.test_bs

        # Train setting
        self.initlr = self.params.initlr  # initial learning rate
        self.weight_decay = self.params.weight_decay  # optimizer weight decay
        self.crop_size = self.params.crop_size  # crop size for data augmentation
        self.num_workers = self.params.num_workers

        # Loss functions
        self.loss_f = nn.MSELoss()  # Replace L1_Charbonnier_loss with MSELoss
        self.loss_l1 = nn.L1Loss()
        self.loss_per = PerceptualLoss2()
        self.model = SnowFormer()

    def forward(self, x, guidance):
        y = self.model(x, guidance)
        return y

    def configure_optimizers(self):
        # Optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initlr, betas=[0.9, 0.999])
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.initlr, max_lr=1.2 * self.initlr,
                                                      cycle_momentum=False)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, guidance, _ = batch
        output = self.forward(x, guidance)

        # Compute losses
        loss_f = self.loss_f(y, output)
        loss_per = self.loss_per(y, output)
        loss = loss_f + 0* loss_per

        # Log the loss
        self.log('train_loss', loss)
        return {'loss': loss}

    def train_dataloader(self):
        # Load training dataset
        train_set = Text(self.train_datasets, train=True, size=self.crop_size)
        train_loader = DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True,
                                  num_workers=self.num_workers)
        return train_loader

    def on_epoch_end(self):
        # Save model checkpoint
        checkpoint_dir = './checkpoints3'
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f'SnowFormer_epoch{self.current_epoch + 1}.pth')
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

# Main function
if __name__ == '__main__':
    args = {
        'epochs': 1000,
        'train_datasets': r'/data1/cxy/Text',
        'test_datasets': r'/data1/cxy/TextTD',
        'train_bs': 1,  # Set batch size to 2
        'test_bs': 40,
        'val_bs': 2,
        'initlr': 0.0006,
        'weight_decay': 0.01,
        'crop_size': 256,
        'num_workers': 16
    }

    ddp = DDPStrategy(find_unused_parameters=True)  # Enable unused parameter detection
    hparams = Namespace(**args)

    model = CoolSystem(hparams)

    trainer = pl.Trainer(
        plugins=[ddp],
        max_epochs=hparams.epochs,
        devices=[0,1],
        accelerator='gpu',
        logger=wandb_logger,
        precision=16,
        gradient_clip_val=1.0
    )

    trainer.fit(model)

