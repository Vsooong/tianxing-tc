import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cyclone_model import CyclonePredictionModel
import torch
import os
import numpy as np


area_encoding = {"ATLN": 0, "EPAC": 1, "WPAC": 2}
vmax_mean = {"ATLN": 50.0, "EPAC": 50.0, "WPAC": 50.0}
vmax_std = {"ATLN": 28.0, "EPAC": 28.0, "WPAC": 28.0}
mslp_mean = {"ATLN": 990, "EPAC": 990, "WPAC": 990}
mslp_std = {"ATLN": 20, "EPAC": 20, "WPAC": 20}
lon_mean = {"ATLN": -60.457963, "EPAC": -120.074932, "WPAC": 132.260497}
lon_std = {"ATLN": 15.00, "EPAC": 15.00, "WPAC": 15.00}
lat_mean = {"ATLN": 26.553046, "EPAC": 17.493838, "WPAC": 19.050167}
lat_std = {"ATLN": 8.0, "EPAC": 6.0, "WPAC": 7.0}
storm_age_mean = 55.03
storm_age_std = 26.69
storm_speed_mean = 18.91
storm_speed_std = 11.59
channel1_mean = 268.9715
channel1_std = 26.70663
channel2_mean = 236.25537
channel2_std = 11.895549


class Profiler(pl.LightningModule):  # pl.LightningModule torch.nn.Module
    def __init__(
        self,
        use_field: bool = False,
        batch_size: int = 32,
    ):
        super(Profiler, self).__init__()
        self.model = CyclonePredictionModel(use_field=use_field)
        self.use_field = use_field
        self.batch_size = batch_size
        self.loss = torch.nn.MSELoss(reduction="none")

    def loss_with_mask(self, output, target, tgt_len):
        mask = torch.zeros_like(output).to(self.device)
        for i in range(len(tgt_len)):
            mask[i, : tgt_len[i]] = 1
        loss = self.loss(output, target) * mask
        if mask.sum() == 0:
            return torch.tensor(0.0).to(self.device)
        return loss.sum() / mask.sum()

    def configure_optimizers(self):

        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        from adabelief_pytorch import AdaBelief

        optimizer = AdaBelief(
            self.model.parameters(),
            lr=1.5e-4,
            eps=1e-16,
            betas=(0.9, 0.999),
            weight_decouple=True,
        )

        # optimizer = torch.optim.AdamW(
        #     [
        #         {"params": self.model.rnn.parameters(), "lr": 1e-4},
        #         {"params": self.model.fc1.parameters(), "lr": 1e-4},
        #         {"params": self.model.fc2.parameters(), "lr": 1e-4},
        #         {"params": self.model.field_branch.parameters(), "lr": 1e-4},
        #         {"params": self.model.image_branch.parameters(), "lr": 1e-4},
        #         {"params": self.model.factor_branch.parameters(), "lr": 1e-4},
        #         {"params": self.model.fusion.parameters(), "lr": 1e-4},
        #     ]
        # )
        step_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.7
        )

        return [optimizer], [step_lr_scheduler]

    def forward(
        self, image, factor, fields=None, pred_lons=None, pred_lats=None, area_id=None
    ):
        return self.model.forward(image, factor, fields, pred_lons, pred_lats, area_id)

    def training_step(self, batch):
        area_id, factors, image, target, tgt_len, pred_field, pred_lons, pred_lats = (
            batch
        )
        if tgt_len.sum() == 0:
            return None  # skip the batch if there is no target

        output = self.forward(image, factors, pred_field, pred_lons, pred_lats, area_id)
        loss = self.loss_with_mask(output, target, tgt_len) * 10
        if isinstance(self, pl.LightningModule):
            self.log(
                "train_loss",
                loss.item(),
                prog_bar=True,
                logger=True,
                # on_epoch=True,
                # on_step=True,
            )

        return loss

    def validation_step(self, batch):
        area_id, factors, image, target, tgt_len, pred_field, pred_lons, pred_lats = (
            batch
        )
        if tgt_len.sum() == 0:
            return None
        output = self.forward(image, factors, pred_field, pred_lons, pred_lats, area_id)
        loss = self.loss_with_mask(output, target, tgt_len) * 20
        if isinstance(self, pl.LightningModule):
            self.log(
                "val_loss",
                loss.item(),
                prog_bar=True,
                logger=True,
            )
        return loss

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
        # val_data = read_tcir(train=False, use_field=self.use_field)
        # val_loader = DataLoader(
        #     val_data,
        #     batch_size=self.batch_size * 2,
        #     shuffle=False,
        #     num_workers=4,
        #     prefetch_factor=2,
        # )
        # return val_loader


def haversine_batch(lon1, lat1, lon2, lat2):
    # lon1 shape (batch_size, 1)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    # 地球半径 (单位：公里)
    r = 6371
    return c * r


def parse_output(target, area):
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()  # (16, 4)

    target[:, 0] = target[:, 0] * vmax_std[area] + vmax_mean[area]
    target[:, 1] = target[:, 1] * mslp_std[area] + mslp_mean[area]
    target[:, 2] = target[:, 2] * lon_std[area] + lon_mean[area]
    target[:, 3] = target[:, 3] * lat_std[area] + lat_mean[area]
    return target


def test(version=1):
    loss_matrix_path = f".../loss_v{version}_matrix.npy"
    checkpoint_path = "..."
    area_code = {0: "ATLN", 1: "EPAC", 2: "WPAC"}
    if not os.path.exists(loss_matrix_path):
        torch.manual_seed(42)
        torch.set_float32_matmul_precision("medium")

        profiler = Profiler(use_field=True, batch_size=16)

        checkpoint = torch.load(checkpoint_path)
        profiler.load_state_dict(checkpoint["state_dict"])

        device = torch.device("cuda:6")
        profiler.device = device
        profiler.model.to(device)
        profiler.loss.to(device)

        profiler.eval()
        loss_maxtrix = np.zeros((3, 16, 5))  # 3 areas, 16 time steps, 5 variables
        num_maxtrix = np.zeros((3, 16, 5))
        with torch.no_grad():
            for batch in profiler.val_dataloader():
                (
                    area_id,
                    factors,
                    image,
                    target,
                    tgt_len,
                    pred_field,
                    pred_lons,
                    pred_lats,
                ) = batch
                if tgt_len.sum() == 0:
                    continue
                image = image.to(device)
                factors = factors.to(device)
                target = target.to(device)
                pred_field = pred_field.to(device)
                pred_lons = pred_lons.to(device)
                pred_lats = pred_lats.to(device)
                area_id = area_id.to(device)

                output = profiler.forward(
                    image, factors, pred_field, pred_lons, pred_lats, area_id
                )
                # mask = torch.zeros_like(output).to(device)
                np_pred = output.cpu().numpy()
                np_target = target.cpu().numpy()

                for i in range(len(tgt_len)):
                    mask = np.zeros((16, 5))
                    areaid = int(area_id[i].item())
                    area = area_code[areaid]
                    vmax_now = factors[i, 0] * 28 + 50
                    vmax_now = vmax_now.cpu().item()

                    if areaid == 2 and vmax_now < 30:
                        continue

                    mask[: tgt_len[i], :] = 1
                    np_target[i] = parse_output(np_target[i], area)
                    np_pred[i] = parse_output(np_pred[i], area)
                    loss = np.abs(np_pred[i] - np_target[i])
                    distance_loss = haversine_batch(
                        np_pred[i, :, 2],
                        np_pred[i, :, 3],
                        np_target[i, :, 2],
                        np_target[i, :, 3],
                    )
                    distance_loss = np.expand_dims(distance_loss, axis=1) / 1.852
                    loss = np.concatenate([loss, distance_loss], axis=1)
                    loss = loss * mask
                    loss_maxtrix[areaid] += loss
                    num_maxtrix[areaid] += mask
            # break
        loss_maxtrix = loss_maxtrix / num_maxtrix
        # save the loss matrix
        np.save(
            loss_matrix_path,
            loss_maxtrix,
        )
