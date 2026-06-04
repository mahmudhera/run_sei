import torch
import torch.nn as nn
from model.sei import Sei
import numpy as np

from utils import load_state_dict_flexible


class FullEntryAttentionMLPHead(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.seq_len = feature_dim * 3

        # Each scalar entry becomes a hidden_dim-dimensional token
        self.value_proj = nn.Linear(1, hidden_dim)

        # Learns which embedding type an entry came from: ref / alt / diff
        self.type_embedding = nn.Parameter(
            torch.randn(1, 3, 1, hidden_dim)
        )

        # Learns the feature index identity: feature 0, feature 1, ...
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1, feature_dim, hidden_dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pool = nn.Sequential(
            nn.Linear(self.seq_len * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ref_emb, alt_emb, diff_emb):
        """
        ref_emb:  [batch, feature_dim]
        alt_emb:  [batch, feature_dim]
        diff_emb: [batch, feature_dim]
        """

        B = ref_emb.size(0)

        x = torch.stack([ref_emb, alt_emb, diff_emb], dim=1)
        # [batch, 3, feature_dim]

        x = x.unsqueeze(-1)
        # [batch, 3, feature_dim, 1]

        x = self.value_proj(x)
        # [batch, 3, feature_dim, hidden_dim]

        x = x + self.type_embedding + self.pos_embedding
        # [batch, 3, feature_dim, hidden_dim]

        x = x.view(B, self.seq_len, self.hidden_dim)
        # [batch, 3 * feature_dim, hidden_dim]

        attn_out, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )
        # [batch, 3 * feature_dim, hidden_dim]

        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        x = x.reshape(B, -1)
        # [batch, 3 * feature_dim * hidden_dim]

        x = self.pool(x)

        out = self.out(x)
        # [batch, 1]

        return out



class AttentionMLPHead(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        self.type_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))
        # token 0 = ref, token 1 = alt, token 2 = diff

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pool = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ref_emb, alt_emb, diff_emb):
        """
        ref_emb:  [batch, feature_dim]
        alt_emb:  [batch, feature_dim]
        diff_emb: [batch, feature_dim]
        """

        x = torch.stack([ref_emb, alt_emb, diff_emb], dim=1)
        # [batch, 3, feature_dim]

        x = self.input_proj(x)
        # [batch, 3, hidden_dim]

        x = x + self.type_embedding
        # lets the model know which token is ref/alt/diff

        attn_out, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            need_weights=True,
        )

        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        x = x.reshape(x.size(0), -1)
        # [batch, hidden_dim * 3]

        x = self.pool(x)
        out = self.out(x)

        return out


class SeiFullPredictor(nn.Module):
    """
    Full Sei model that returns the 21,907 chromatin-profile predictions.
    """
    def __init__(self, pretrained_path, device="cuda"):
        super().__init__()
        self.model = Sei(sequence_length=4096, n_genomic_features=21907)

        state = load_state_dict_flexible(pretrained_path, map_location=device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)

        self.model = self.model.to(device)

        if missing:
            print("WARNING: missing keys (showing up to 20):", missing[:20])
        if unexpected:
            print("WARNING: unexpected keys (showing up to 20):", unexpected[:20])

    def forward(self, x):
        return self.model(x)   # [B, 21907]



class VariantEffectModel(nn.Module):
    def __init__(self, pretrained_path, hidden_dim=512, freeze_backbone=True, device="cuda"):
        super().__init__()

        self.backbone = SeiFullPredictor(pretrained_path, device=device)
        device = torch.device(device)
        print(f"Moving model to device: {device}")
        self.backbone = self.backbone.to(device)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # extract the feature dimension from the backbone
        dummy_input = torch.randn(1, 4, 4096)  # [B, C, L]
        dummy_input = dummy_input.to(device)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)
        feature_dim = backbone_output.size(1)

        print(f"Backbone feature dimension: {feature_dim}")
        print(f"Head hidden dimension: {hidden_dim}")

        # project to sequence classes using model/projvec_targets.npy
        proj_matrix = np.load("model/projvec_targets.npy").astype(np.float32)  # [x, 21907]
        proj_matrix = torch.from_numpy(proj_matrix).to(device)  # [x, 21907]
        # make [x, 21907] -> [21907, x]
        proj_matrix = proj_matrix.t()  # [21907, x]
        self.proj_matrix = nn.Parameter(proj_matrix, requires_grad=not freeze_backbone)

        dim_after_proj = proj_matrix.size(1)
        print(f"Dimension after projection: {dim_after_proj}")

        # send proj matrix to same device as backbone
        self.proj_matrix = self.proj_matrix.to(device)


        self.head = FullEntryAttentionMLPHead(
            feature_dim=dim_after_proj,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
        )
        
        self.head = self.head.to(device)

    def forward(self, ref, alt):
        ref_feat = self.backbone(ref)
        alt_feat = self.backbone(alt)
        diff = alt_feat - ref_feat

        feature_dim = ref_feat.size(1)
        ref_emb = ref_feat.view(-1, feature_dim)
        alt_emb = alt_feat.view(-1, feature_dim)
        diff_emb = diff.view(-1, feature_dim)

        # Apply projection matrix
        ref_emb = torch.matmul(ref_emb, self.proj_matrix)
        alt_emb = torch.matmul(alt_emb, self.proj_matrix)
        diff_emb = torch.matmul(diff_emb, self.proj_matrix)

        out = self.head(ref_emb, alt_emb, diff_emb)
        return out.squeeze(1)