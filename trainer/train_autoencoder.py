from pathlib import Path

import torch
import numpy as np
import hydra
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from dataset.distance_field import DistanceFieldDataset
from model.autoencoder import AutoEncoder32
from trainer import create_trainer
from util.df_metrics import IoU, Chamfer3D, Precision, Recall


class AutoencoderTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.train_data = DistanceFieldDataset(config, interval=[0, -1024])
        self.val_data = DistanceFieldDataset(config, interval=[-1024, None])
        self.num_vis_samples = 48
        self.model = AutoEncoder32()
        self.metrics = torch.nn.ModuleList([IoU(compute_on_step=False), Chamfer3D(compute_on_step=False),
                                            Precision(compute_on_step=False), Recall(compute_on_step=False)])

    def configure_optimizers(self):
        opt = torch.optim.Adam(list(self.model.parameters()), lr=self.config.lr_e, eps=1e-8, weight_decay=1e-4)
        return opt

    def forward(self, batch):
        return self.model(self.normalize_df(batch['df']))

    def training_step(self, batch, batch_idx):
        self.train_data.augment_batch_data(batch)
        predicted = self.network_pred_to_df(self.forward(batch))
        weights = self.adjust_weights(predicted >= self.train_data.trunc, batch)
        loss_l1 = (torch.abs(predicted - batch['df']) * weights).mean()
        self.log("train/l1", loss_l1, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss_l1

    def validation_step(self, batch, batch_idx):
        predicted = self.network_pred_to_df(self.forward(batch))
        loss_l1 = self.record_evaluation_for_batch(predicted, batch)
        self.log("val/l1", loss_l1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if batch_idx < self.num_vis_samples:
            output_dir = Path(f'runs/{self.config.experiment}/visualization/{self.current_epoch:04d}')
            output_dir.mkdir(exist_ok=True, parents=True)
            for b in range(predicted.shape[0]):
                self.val_data.visualize_as_mesh(predicted[b][0].cpu().numpy(), output_dir / f"{batch['name'][b]}_pred.obj")
                self.val_data.visualize_as_mesh(batch['df'][b][0].cpu().numpy(), output_dir / f"{batch['name'][b]}_gt.obj")

    @rank_zero_only
    def validation_epoch_end(self, _outputs):
        iou, cd, precision, recall = self.metrics[0].compute(), self.metrics[1].compute(), self.metrics[2].compute(), self.metrics[3].compute()
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/cd", cd, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/precision", precision, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/recall", recall, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        print(f"\nIoU = {iou:.3f} | CD = {cd:.3f} | P = {precision:.3f} | R = {recall:.3f}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.config.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=self.config.num_workers)

    def test_dataloader(self):
        data = DistanceFieldDataset(self.config, interval=[0, None])
        return torch.utils.data.DataLoader(data, batch_size=self.config.batch_size, shuffle=False, pin_memory=False, drop_last=False, num_workers=self.config.num_workers)

    @staticmethod
    def adjust_weights(pred_empty, batch):
        weights = batch['weights'].clone().detach()
        weights[batch['empty'] & pred_empty] = 0
        return weights

    def network_pred_to_df(self, clamped_out):
        return (clamped_out + 1) * self.train_data.trunc / 2

    def normalize_df(self, df):
        return self.train_data.normalize(df)

    def record_evaluation_for_batch(self, pred_shape_df, batch):
        target_shape = batch['df'] <= self.train_data.vox_size
        predicted_shape = pred_shape_df <= self.train_data.vox_size
        for metric in self.metrics:
            metric(predicted_shape, target_shape)
        loss_l1 = torch.abs(pred_shape_df - batch['df']).mean()
        return loss_l1

    def test_step(self, batch, batch_idx):
        predicted = self.network_pred_to_df(self.forward(batch))
        code = self.model.encoder(self.normalize_df(batch['df']))
        b, c, d, h, w = code.shape
        code = code.reshape(b, c * d * h * w).cpu().numpy()
        self.record_evaluation_for_batch(predicted, batch)
        output_dir = Path(f'runs/{self.config.experiment}/latent/{self.current_epoch:04d}')
        output_dir.mkdir(exist_ok=True, parents=True)
        for b in range(code.shape[0]):
            np.save(output_dir / f'{batch["name"][b]}.npy', code[b])

    def test_epoch_end(self, outputs):
        iou, cd, precision, recall = self.metrics[0].compute(), self.metrics[1].compute(), self.metrics[2].compute(), self.metrics[3].compute()
        print(f"\nIoU = {iou:.3f} | CD = {cd:.3f} | P = {precision:.3f} | R = {recall:.3f}")


@hydra.main(config_path='../config', config_name='stylegan2')
def main(config):
    trainer = create_trainer("Autoencoder32", config)
    model = AutoencoderTrainer(config)
    trainer.fit(model)


@hydra.main(config_path='../config', config_name='stylegan2')
def infer(config):
    trainer = create_trainer("Autoencoder32", config)
    model = AutoencoderTrainer(config)
    model.load_state_dict(torch.load(config.resume)['state_dict'])
    trainer.test(model)


if __name__ == '__main__':
    main()
