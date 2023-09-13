from os import path
import json
import math
from argparse import Namespace
from typing import Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.functional as F
import torch.utils.checkpoint as checkpoint


from esm import pretrained
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.inverse_folding.gvp_transformer_encoder import GVPTransformerEncoder
from esm.model.esm2 import ESM2
from esm.data import Alphabet
from esm.inverse_folding.util import nan_to_num, get_rotation_frames, rotate, rbf
from esm.inverse_folding.features import GVPInputFeaturizer

SAVE_DIR = path.join(path.dirname(path.abspath(__file__)), "config")
DEFAULT_JOINT_EMBEDDING_DIM = 512
DEFAULT_POOL_LAST_N_LAYERS = 4
DEFAULT_TOKEN_POOL_STRATEGY = "bos"
DEFAULT_LAYER_POOL_STRAEGY = "weighted"


def generate_alphabet_args(model="esm2"):
    """
    Generate args dict for ESM2 Alphabet from Hub.

    Args:
        model (str, optional): Model name: esm2/esm_if. Defaults to "esm2".
        save (bool, optional): Save args to json file. Defaults to True.
    """
    assert model in ["esm2", "esm_if"], "Model Name must be esm2 or esm_if"
    if model == "esm2":
        _, alphabet = pretrained.esm2_t6_8M_UR50D()
    else:
        _, alphabet = pretrained.esm_if1_gvp4_t16_142M_UR50()
    args = {
        "standard_toks": alphabet.standard_toks,
        "prepend_toks": alphabet.prepend_toks,
        "append_toks": alphabet.append_toks,
        "prepend_bos": alphabet.prepend_bos,
        "append_eos": alphabet.append_eos,
        "use_msa": alphabet.use_msa,
    }

    return args


def generate_esm_if_args():
    model, _ = pretrained.esm_if1_gvp4_t16_142M_UR50()
    model_args = vars(model.args)
    return model_args


# TODO: revisit this.
def visualize_logits(self, layers: list) -> None:
    """Visualize Logits

    Args:
        layers (list): List of tuples of name and tensor containing logits
            eg: [("esm2_outs", esm2_outs), ("esm_if_outs", esm_if_outs)]
    """
    import matplotlib.pyplot as plt
    import torch

    # visualize histograms
    plt.figure(figsize=(20, 4))  # width and height of the plot
    legends = []
    for i, layer in enumerate(layers):  # note: exclude the output layer
        #   if isinstance(layer, Tanh):
        t = layer[1]
        print(
            "layer {%d (%10s)}: mean %+.2f, std %.2f" % (i, layer[0], t.mean(), t.std())
        )
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer[0]}")
    plt.legend(legends)
    plt.title("activation distribution")


class SequenceEncoder(ESM2):
    """Modified ESM2 Class with pooling and joint embedding projections"""

    def __init__(self, args: Namespace):
        """Initialize Modified ESM2 Model inherited from esm.model.esm2.ESM2

        Args:
            args (Namespace): Args containing model params / config
                - num_layers (int): Number of Layers
                - embed_dim (int): Embedding Dimension
                - attention_heads (int): Number of Attention Heads
                - token_dropout (bool): Whether to apply token dropout
                - layer_pool_strategy (str): Layer Pooling Strategy (weighted/attention)
                - pool_last_n_layers (int): Number of Hidden Layers to pool
                - token_pool_strategy (str): Token Pooling Strategy (mean/bos)
                - layer_pool_weights (torch.tensor): Layer weights for weighted layer pooling
                - joint_embedding_dim (int): Joint Embedding Dimension
                - alphabet (Union[Alphabet, str], optional): ESM2 Alphabet. Defaults to "ESM-1b".
        """

        super().__init__(
            args.num_layers,
            args.embed_dim,
            args.attention_heads,
            args.alphabet,
            args.token_dropout,
        )

        # Remove layers not used in loss calculation
        del self.lm_head
        del self.contact_head

        # Token Pooler
        self.token_pool_strategy = args.token_pool_strategy
        assert self.token_pool_strategy in ["mean", "bos"]

        # Layer Pooler
        self.layer_pool_strategy = args.layer_pool_strategy
        if self.layer_pool_strategy is not None:
            self.pool_last_n_layers = args.pool_last_n_layers
            assert (
                args.pool_last_n_layers <= args.num_layers
            ), "pool_last_n_layers must be less than or equal to than num_layers"

            self.layer_pooler = get_layer_pooler(
                self.layer_pool_strategy,
                self.pool_last_n_layers,
                args.layer_pool_weights,
            )

        # Project Sequence Embedding to Joint Embedding Space
        self.joint_embedding_projection = nn.Sequential(
            nn.Linear(args.embed_dim, 4 * args.joint_embedding_dim),
            nn.Tanh(),
            nn.Linear(4 * args.joint_embedding_dim, args.joint_embedding_dim),
        )
        self.after_proj_ln = nn.LayerNorm(args.joint_embedding_dim)
        self.after_proj_dropout = nn.Dropout(args.final_layer_dropout)
        self.activation_checkpointing = args.activation_checkpointing

    def forward(self, tokens: torch.tensor):
        """
        Forward Function for Seq Encoder
        """
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)

        x = self.embed_scale * self.embed_tokens(tokens)

        # Dropout
        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(
                x.dtype
            ) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        if self.activation_checkpointing:
            for layer in self.layers:
                x, _ = checkpoint.checkpoint(
                    layer.forward, x, None, padding_mask, False
                )
        else:
            for layer in self.layers:
                x, _ = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=False,
                )

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)

        if self.token_pool_strategy == "mean":
            if padding_mask is None:
                B, T, E = x.shape
                batch_padding_lens = torch.tensor(T).expand(B)
            else:
                batch_padding_lens = (~padding_mask).sum(-1)
            pool_tokens = return_mean_of_token_embeddings(x, batch_padding_lens)
        elif self.token_pool_strategy == "bos":
            pool_tokens = x[:, 0]

        # Joint Embedding Projections & Regularizations
        seq_projection = self.joint_embedding_projection(pool_tokens)
        seq_projection = self.after_proj_dropout(seq_projection)
        seq_projection = self.after_proj_ln(seq_projection)
        seq_projection = normalize_embeddings(seq_projection)

        return seq_projection

    def freeze_layers(self, first_n_layers: int):
        """Freeze first n layers of Sequence Encoder

        Args:
            first_n_layers (int): Number of layers to freeze
        """
        assert first_n_layers <= self.num_layers, "first_n_layers must be <= num_layers"
        # freeze embed_tokens layer
        self.embed_tokens.weight.requires_grad = False

        # freeze first n layers
        for i in range(first_n_layers):
            for param in self.layers[i].parameters():
                param.requires_grad = False
        print(f"Froze first {first_n_layers} layers of sequence encoder")


class StructureEncoder(GVPTransformerEncoder):
    def __init__(self, args: Namespace):
        """Initialize ESM_IF Model inherited from GVPTransformerEncoder

        Args:
            args (argparse.Namespace): GVPTransformerEncoder Args. Other args:
                - layer_pool_strategy (str): Layer Pooling Strategy (weighted/attention)
                - pool_last_n_layers (int): Number of Hidden Layers to pool
                - token_pool_strategy (str): Token Pooling Strategy (mean/bos)
                - joint_embedding_dim (int): Joint Embedding Dimension
                - layer_pool_weights (torch.tensor): Layer weights for weighted layer pooling
            alphabet (Alphabet): ESM-IF Alphabet
        """
        encoder_embed_tokens = self.build_embedding(
            args.alphabet, args.encoder_embed_dim
        )
        super().__init__(args, args.alphabet, encoder_embed_tokens)

        # Token Pool
        self.token_pool_strategy = args.token_pool_strategy
        assert self.token_pool_strategy in ["mean", "bos"]

        # Layer Pool
        self.layer_pool_strategy = args.layer_pool_strategy
        if self.layer_pool_strategy is not None:
            self.pool_last_n_layers = args.pool_last_n_layers
            self.num_layers = args.encoder_layers
            assert (
                args.pool_last_n_layers <= self.num_layers
            ), "pool_last_n_layers must be less than or equal to than num_layers"

            self.layer_pooler = get_layer_pooler(
                self.layer_pool_strategy,
                self.pool_last_n_layers,
                args.layer_pool_weights,
            )

        # Joint Embedding Projection
        self.joint_embedding_projection = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, 4 * args.joint_embedding_dim),
            nn.Tanh(),
            nn.Linear(4 * args.joint_embedding_dim, args.joint_embedding_dim),
        )
        self.after_proj_ln = nn.LayerNorm(args.joint_embedding_dim)
        self.after_proj_dropout = nn.Dropout(args.final_layer_dropout)
        self.activation_checkpointing = args.activation_checkpointing

    @classmethod
    def build_embedding(cls, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim**-0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward_embedding_optimized(self, coords, padding_mask, confidence):
        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        coords = nan_to_num(coords)

        mask_tokens = (
            padding_mask * self.dictionary.padding_idx
            + ~padding_mask * self.dictionary.get_idx("<mask>")
        )
        # tokens
        x = self.embed_tokens(mask_tokens) * self.embed_scale

        # dihedrals
        x = x + self.embed_dihedrals(coords)

        if self.activation_checkpointing:
            gvp_out_scalars, gvp_out_vectors = checkpoint.checkpoint(
                self.gvp_encoder.forward, coords, coord_mask, padding_mask, confidence
            )

        else:
            gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(
                coords, coord_mask, padding_mask, confidence
            )

        R = get_rotation_frames(coords)

        x = x + self.embed_gvp_output(
            torch.cat(
                [
                    gvp_out_scalars,
                    rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
                ],
                dim=-1,
            )
        )
        x = x + self.embed_confidence(rbf(confidence, 0.0, 1.0))

        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            coords, coord_mask, with_coord_mask=False
        )
        features = torch.cat(
            [
                scalar_features,
                rotate(vector_features, R.transpose(-2, -1)).flatten(-2, -1),
            ],
            dim=-1,
        )

        x = self.dropout_module(
            x
            + self.embed_gvp_input_features(features)
            + self.embed_positions(mask_tokens)
        )
        return x

    def forward(
        self,
        coords: torch.tensor,
        encoder_padding_mask: torch.tensor,
        confidence: torch.tensor,
    ) -> torch.tensor:
        """Forward function

        Args:
            coords (torch.tensor): Coords
            encoder_padding_mask (torch.tensor): Padding Mask
            confidence (torch.tensor): Confidence

        Returns:
            torch.tensor: Structure Representation Pooled.
        """
        x = self.forward_embedding_optimized(coords, encoder_padding_mask, confidence)
        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        x = self.layer_norm(x)
        x = x.transpose(0, 1)  # B * T * C

        if self.token_pool_strategy == "mean":
            batch_padding_lens = (~encoder_padding_mask).sum(-1)
            pool_tokens = return_mean_of_token_embeddings(x, batch_padding_lens)
        else:
            pool_tokens = x[:, 0]  # B * C

        # Joint Embedding Projections & Regularizations
        structure_embedding = self.joint_embedding_projection(pool_tokens)
        structure_embedding = self.after_proj_dropout(structure_embedding)
        structure_embedding = self.after_proj_ln(structure_embedding)
        structure_embedding = normalize_embeddings(structure_embedding)
        return structure_embedding

    def freeze_layers(self, first_n_layers: int):
        """Freeze first n layers of Sequence Encoder

        Args:
            first_n_layers (int): Number of layers to freeze
        """
        assert first_n_layers <= len(
            self.layers
        ), "first_n_layers must be <= len(self.layers)"

        # freeze all layers before self.layers
        for name, param in self.named_parameters():
            if "encoder_layers" in name:
                param.requires_grad = False
            if "layers." not in name:
                param.requires_grad = False
            else:
                break

        # freeze the first n layers in self.layers
        for i in range(first_n_layers):
            for param in self.layers[i].parameters():
                param.requires_grad = False
        print(f"Froze first {first_n_layers} layers of structure encoder")


class WeightedLayerPooling(nn.Module):
    """Implementation of Weighted Layer Pooling Taken from
    https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently

    """

    def __init__(
        self,
        pool_last_n_layers: int,
        layer_weights=None,
    ):
        """Init Params

        Args:
            pool_last_n_layers (int): Number of Hidden Layers in the Model
            layer_start (int): Take the last n layers for weight pooling
            layer_weights (_type_, torch.tensor): Layer weights. Defaults to None.
            pool_strategy (str): Pooling Strategy (mean/bos/eos).
                mean: take a mean of all token embeddings in a batch.
                bos: take the first token embedding in a batch.
                eos: take the last token embedding in a batch.
        """
        super().__init__()
        self.layer_weights = nn.Parameter(torch.tensor([1.0] * (pool_last_n_layers)))

    def forward(
        self,
        pooled_encoder_states: torch.tensor,
    ) -> torch.tensor:
        """Forward Function for Weighted Layer Pooling

        Args:
            pooled_encoder_states (torch.tensor): All hidden states pooled per layer of size
                Num Hidden Layers * Batch Size * Emb Size. Should be called after per_layer pooling

        Returns:
            torch.tensor: weighted average of all hidden states
        """
        weight_factor = (
            self.layer_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(pooled_encoder_states.size())
        )
        weighted_average = (weight_factor * pooled_encoder_states).sum(
            dim=0
        ) / self.layer_weights.sum()
        return weighted_average


class AttentionPooling(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, out_dim: int):
        super(AttentionPooling, self).__init__()
        self.pool_last_n_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(
            loc=0.0, scale=0.1, size=(self.hidden_size, self.out_dim)
        )
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()

    def forward(self, all_hidden_states):
        hidden_states = torch.stack(
            [
                all_hidden_states[layer_i][:, 0].squeeze()
                for layer_i in range(0, self.pool_last_n_layers)
            ],
            dim=-1,
        )
        hidden_states = hidden_states.view(
            -1, self.pool_last_n_layers, self.hidden_size
        )
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


def get_layer_pooler(
    layer_pool_strategy: str,
    pool_last_n_layers: int,
    layer_weights: torch.tensor = None,
) -> nn.Module:
    """Get Pooler

    Args:
        layer_pool_strategy (str): Pooler Type (weighted/attention)
        num_hidden_layers (int): Number of Hidden Layers to pool
        pool_strategy (str): Pooling Strategy (mean/bos).
        layer_weights (None, torch.tensor): Layer weights. Defaults to None.

    Returns:
        nn.Module: Respective Pooler Module
    """
    assert layer_pool_strategy in [
        "weighted",
        "attention",
    ], "layer_pool_strategy must be weighted/attention"
    if layer_pool_strategy == "weighted":
        return WeightedLayerPooling(pool_last_n_layers, layer_weights)
    else:
        return AttentionPooling(pool_last_n_layers, 1024, 1024)


def return_mean_of_token_embeddings(
    token_embeddings: torch.tensor, batch_padding_lens: torch.tensor
) -> torch.tensor:
    """
    Take mean of token embeddings from encoder states.
    Collapses the sequence dimension.


    Args:
        token_embeddings (torch.tensor): Encoder States of shape Hidden Layers x Batch Size x Seq Len x Emb Size.
        batch_padding_lens (torch.tensor): Batch Padding lengths of length: Batch Size.

    Returns:
        torch.tensor: Output of shape (num_hidden_layers, batch_size, emb_size)
    """
    assert token_embeddings.ndim == 3
    B, T, E = token_embeddings.shape

    pooled_token_embeddings = torch.zeros(B, E, device=token_embeddings.device)
    for i, tokens_len in enumerate(batch_padding_lens):
        if tokens_len <= 2:
            pooled_token_embeddings[i] = token_embeddings[i, 1]
        else:
            pooled_token_embeddings[i] = token_embeddings[i, 1 : tokens_len - 1].mean(
                dim=0
            )
    return pooled_token_embeddings


def normalize_embeddings(embeddings: torch.tensor) -> torch.tensor:
    """Normalize Embeddings
        Taken from https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#L362

    Args:
        embeddings (torch.tensor): Embedding. (B, E)

    Returns:
        torch.tensor: Normalized Embeddings. (B, E)
    """
    return embeddings / embeddings.norm(dim=1, keepdim=True)


def load_sequence_encoder(
    model_config: dict,
) -> Tuple[SequenceEncoder, Alphabet]:
    """Load ESM2 model using saved args

    Args:
        model_config (dict): Model Config
        joint_embedding_dim (int): Joint Embedding Dimension

    Returns:
        Tuple[SequenceEncoder, Alphabet]: ESM2 model and Alphabet
    """
    # Load ESM-2 Base model args, alphabet args, and the given model config

    with open(path.join(SAVE_DIR, "default_alphabet_args.json"), "r") as f:
        alphabet_args = json.load(f)["esm2"]

    alphabet = Alphabet(**alphabet_args)
    model_config["alphabet"] = alphabet
    seq_encoder = SequenceEncoder(args=Namespace(**model_config))

    return seq_encoder, alphabet


def load_structure_encoder(
    model_config: dict,
) -> Tuple[StructureEncoder, Alphabet]:
    """Load ESM-IF model using saved args

    Args:
        model_config (dict): ESM-IF Args

    Returns:
        Tuple[StructureEncoder, Alphabet]: ESM-IF model and Alphabet
    """

    with open(path.join(SAVE_DIR, "default_alphabet_args.json"), "r") as f:
        alphabet_args = json.load(f)["esm_if"]

    with open(path.join(SAVE_DIR, "default_esm_if_args.json"), "r") as f:
        esm_if_args = json.load(f)

    # Update model args with the given model type args
    for arg, val in model_config.items():
        esm_if_args[arg] = val

    alphabet = Alphabet(**alphabet_args)
    esm_if_args["alphabet"] = alphabet

    structure_encoder = StructureEncoder(Namespace(**esm_if_args))

    return structure_encoder, alphabet


class WarmupCosineFactorLambda(object):
    def __init__(
        self,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        final_lr: float = 0.0,
        eps: int = 1e-8,
    ) -> None:
        """Holds the lamda function for CosineSchedule with Warmup
        for torch.optim.lr_scheduler.LambdaLR

        Args:
            warmup_steps (int): Warmup steps
            max_lr (float): Start Learning Rate
            max_steps (int): Max iterations per epoch
            final_lr (float, optional): Final Learning Rate. Defaults to 0.0.
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        assert (
            self.warmup_steps < self.max_steps
        ), "Warmup steps must be less than max_steps"

        # self.max_lr = max_lr
        self.final_lr_factor = final_lr / max_lr
        self.eps = eps

    def compute_lr_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            progress = float(step) / float(max(1, self.warmup_steps))
            lr_factor = progress + self.eps
        else:
            if int(step / self.max_steps) < 1:
                progress = float(step - self.warmup_steps) / float(
                    max(1, self.max_steps - self.warmup_steps)
                )
            else:
                progress = float(step) / float(max(1, self.max_steps))
            lr_factor = max(
                self.final_lr_factor,
                self.final_lr_factor
                + 0.5
                * (1 - self.final_lr_factor)
                * (1.0 + (math.cos(math.pi * progress))),
            )
        return lr_factor
