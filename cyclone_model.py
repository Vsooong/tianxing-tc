import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# no warning
import warnings

warnings.filterwarnings("ignore")


class CyclonePredictionModel(torch.nn.Module):
    def __init__(
        self,
        field_dim=72,
        image_dim=2,
        factor_dim=18,
        hiddim=560,
        use_field=False,
        dropout_rate=0.2,
    ):
        super(CyclonePredictionModel, self).__init__()
        self.image_branch = Encoder_image(image_dim, hiddim)
        self.factor_branch = nn.Sequential(
            nn.Linear(factor_dim, 128),
            nn.Linear(128, hiddim),
            nn.Tanh(),
        )
        if use_field:
            self.field_branch = Encoder_field(field_dim)
            # self.fc_field = nn.Linear(hiddim + 1, 2)
            self.fusion = ConcatFusion(hiddim * 3, hiddim)
            self.rnn = nn.GRU(
                input_size=hiddim,
                hidden_size=hiddim,

                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            # self.layer_norm = nn.LayerNorm(hiddim + 2)
            self.act = nn.SiLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(hiddim, 128)
            self.fc2 = nn.Linear(130, 4)
            # self.fc1 = nn.Linear(592, 64)

        else:
            self.field_branch = None
            self.fusion = ConcatFusion(hiddim * 2, hiddim)
            self.fc_image = nn.Linear(hiddim, 2)
            self.fc_intermediate = nn.Linear(hiddim, 64)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # 根据需要调整bias的初始化
                    param.data[: param.size(0) // 3].fill_(0.1)

    def forward(
        self, images, factors, fields=None, pred_lons=None, pred_lats=None, area_id=None
    ):
        """
        Args:
            fields: the predicted fields, shape is (batch_size, 17, 70, 50, 50)
            images: the remote sensing images, shape is (batch_size, 2, 101, 101)
            factors: the statistical factors, shape is (batch_size, 18)
            pred_lons: the predicted longitudes, shape is (batch_size, 17)
            pred_lats: the predicted latitudes, shape is (batch_size, 17)

        Returns:
            output: the prediction of the intensity and path, shape is (batch_size,sequence_length, 4)
            intensity: the prediction of the intensity, shape is (batch_size, 2)
        """
        factor_features = self.factor_branch(factors)
        image_features = self.image_branch(images)

        if self.field_branch is not None:
            b, t, c, h, w = fields.size()
            fields = fields.view(b * t, c, h, w)
            field_features = self.field_branch(fields)
            field_features = field_features.view(b, t, -1)
            inital_field = field_features[:, 0, :]
            # inital_intensity = self.fc_field(torch.cat([inital_field, area_id], dim=1))
            feature_list = [inital_field, image_features, factor_features]
            combined_features = self.fusion(feature_list)
            combined_features = combined_features.unsqueeze(0)
            pred_track = torch.stack([pred_lons, pred_lats], dim=2)[:, 1:, :]
            gru_out, _ = self.rnn(field_features[:, 1:, :], combined_features)
            # gru_out = self.layer_norm(self.act(gru_out))
            output = self.dropout(gru_out)
            output = self.fc1(self.act(output))
            output = torch.cat([output, pred_track], dim=2)
            output = self.dropout(output)
            output = self.fc2(output)
            return output

            # _,gru_out = self.rnn(field_features[:, 1:, :], combined_features)
            # output = self.dropout(self.act(gru_out.squeeze(0)))
            # pred_track=torch.flatten(pred_track, start_dim=1)
            # output = torch.cat([output, pred_track], dim=1)
            # output = self.fc1(output)
            # output = output.view(
            #     b, t-1, 4
            # )
            # return output

        else:
            batch_size = images.size(0)
            feature_list = [image_features, factor_features]
            inital_intensity = self.fc_image(image_features)
            combined_features = self.fusion(feature_list)
            intermediate_output = self.fc_intermediate(self.dropout(combined_features))
            intermediate_output = intermediate_output.view(
                batch_size, self.sequence_length, 4
            )
            return intermediate_output, inital_intensity


class Encoder_field(nn.Module):
    def __init__(self, in_channels, hiddim=280, out_channels=560):
        super(Encoder_field, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hiddim, 3, 1, 0)
        self.attention = TAUSubBlock(hiddim)
        self.conv2 = nn.Conv2d(hiddim, hiddim, 5, 2, 0)
        self.conv3 = nn.Conv2d(hiddim, hiddim, 3, 1, 0)
        self.act = nn.SiLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.AdaptiveMaxPool2d(3)
        self.fc1 = nn.Linear(hiddim * 3 * 3, out_channels)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.attention(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class Encoder_image(nn.Module):
    def __init__(self, in_channels, out_channels=560):
        super(Encoder_image, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(512)
        self.act = nn.SiLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.pool_extra = nn.AdaptiveMaxPool2d(2)
        self.fc1 = nn.Linear(512 * 2 * 2, out_channels)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.norm1(x)
        x = self.pool(self.act(x))
        x = self.conv4(self.conv3(x))
        x = self.norm2(x)
        x = self.pool_extra(self.act(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class ConcatFusion(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(ConcatFusion, self).__init__()
        self.act = nn.SELU()
        self.fc = nn.Linear(input_dim, feature_dim)

    def forward(self, feature_list):

        feature_list = [self.act(x) for x in feature_list]
        x = torch.cat(feature_list, dim=1)
        x = self.fc(x)
        return x  


class TransformerFusion(nn.Module):
    def __init__(
        self, feature_dim, num_layers=1, nhead=1, dim_feedforward=560, dropout=0.1
    ):
        super(TransformerFusion, self).__init__()
        encoder_layers = TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, combined_features):
        combined_features = self.norm(combined_features)
        transformer_output = self.transformer_encoder(
            combined_features
        )  # Shape: (batch_size, 3, feature_dim)
        # Aggregate the outputs
        # aggregated_features = transformer_output.mean(
        #     dim=1
        # )  # Shape: (batch_size, feature_dim)
        return transformer_output


from torchvision import models


class ImageBranch(nn.Module):
    def __init__(self):
        super(ImageBranch, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)

    def forward(self, x):
        x = self.resnet(x)
        return x


class TAUSubBlock(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size=21,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.1,
        init_value=1e-2,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = TemporalAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"layer_scale_1", "layer_scale_2"}

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        return x


class TemporalAttention(nn.Module):

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.activation = nn.GELU()  # GELU
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class TemporalAttentionModule(nn.Module):

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False),  # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False),  # expansion
            nn.Sigmoid(),
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)  # depth-wise conv
        attn = self.conv_spatial(attn)  # depth-wise dilation convolution
        f_x = self.conv1(attn)  # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u


class MixMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = DWConv(hidden_features)  # CFF: Convlutional feed-forward network
        self.act = act_layer()  # GELU
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)  # 1x1
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=0,
        dilation=1,
        downsampling=False,
        act_norm=False,
        act_inplace=True,
    ):
        super(BasicConv2d, self).__init__()
        stride = 2 if downsampling is True else 1
        self.act_norm = act_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


if __name__ == "__main__":
    pass
    # area_id = torch.rand(2, 1)
    # field = torch.rand(2, 17, 72, 50, 50)
    # image = torch.rand(2, 2, 101, 101)
    # factor = torch.rand(2, 18)
    # pred_lons = torch.rand(2, 17)
    # pred_lats = torch.rand(2, 17)
    # model = CyclonePredictionModel(use_field=True)
    # output = model(image, factor, field, pred_lons, pred_lats, area_id)
    # print(output.shape)
    # print(sum(p.numel() for p in model.parameters()))
    

    
