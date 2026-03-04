import torch
import torch.nn as nn
from model.sei import Sei

from utils import load_state_dict_flexible


class SeiBackbone(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.model = Sei(sequence_length=4096, n_genomic_features=21907)
        state = load_state_dict_flexible(args.pretrained, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)

        if missing:
            print("WARNING: missing keys (showing up to 20):", missing[:20])
        if unexpected:
            print("WARNING: unexpected keys (showing up to 20):", unexpected[:20])
        
        self.model.classifier = nn.Identity()  # remove classifier

    def forward(self, x):
        # Forward until spline layer
        lout1 = self.model.lconv1(x)
        out1 = self.model.conv1(lout1)

        lout2 = self.model.lconv2(out1 + lout1)
        out2 = self.model.conv2(lout2)

        lout3 = self.model.lconv3(out2 + lout2)
        out3 = self.model.conv3(lout3)

        dconv_out1 = self.model.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.model.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.model.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.model.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.model.dconv5(cat_out4)
        out = cat_out4 + dconv_out5

        spline_out = self.model.spline_tr(out)
        reshape_out = spline_out.view(spline_out.size(0), -1)

        return reshape_out  # 960 * df


class VariantEffectModel(nn.Module):
    def __init__(self, pretrained_path, hidden_dim=512, freeze_backbone=True):
        super().__init__()

        self.backbone = SeiBackbone(pretrained_path)

        feature_dim = 960 * self.backbone.model._spline_df

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ref, alt):
        ref_feat = self.backbone(ref)
        alt_feat = self.backbone(alt)

        combined = torch.cat([ref_feat, alt_feat], dim=1)
        out = self.head(combined)

        return out.squeeze(1)