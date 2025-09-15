from functools import partial

import torch
import transformers.models.convbert as c_bert
from torch import nn


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        """
        Applies global max pooling over timesteps dimension
        """

        super().__init__()
        self.global_max_pool1d = partial(torch.max, dim=1)

    def forward(self, x):
        out, _ = self.global_max_pool1d(x)
        return out


class GlobalAvgPooling1D(nn.Module):
    def __init__(self):
        """
        Applies global average pooling over timesteps dimension
        """

        super().__init__()
        self.global_avg_pool1d = partial(torch.mean, dim=1)

    def forward(self, x):
        out = self.global_avg_pool1d(x)
        return out


class BaseModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str | None = None,
    ):
        """
        Base model that consists of ConvBert layer.

        Args:
            input_dim: Dimension of the input embeddings.
            nhead: Integer specifying the number of heads for the `ConvBert` model.
            hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
            nlayers: Integer specifying the number of layers for the `ConvBert` model.
            kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
            pooling: String specifying the global pooling function. Accepts "avg" or "max". Default: "max"
        """
        super().__init__()

        self.model_type = "Transformer"
        encoder_layers_Config = c_bert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=kernel_size,
            num_hidden_layers=num_hidden_layers,
            hidden_dropout_prob=dropout,
        )

        self.transformer_encoder = nn.ModuleList(
            [c_bert.ConvBertLayer(encoder_layers_Config) for _ in range(num_layers)]
        )

        if pooling is not None:
            if pooling in {"avg", "mean"}:
                self.pooling = GlobalAvgPooling1D()
            elif pooling == "max":
                self.pooling = GlobalMaxPooling1D()
            else:
                raise ValueError(f"Expected pooling to be [`avg`, `max`]. Recieved: {pooling}")

    def convbert_forward(self, x, attention_mask):
        for convbert_layer in self.transformer_encoder:
            x = convbert_layer(x, attention_mask)[0]
        return x


class ConvBertForHead(BaseModule):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super(ConvBertForHead, self).__init__(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=None,
        )
        """
            ConvBert model for multiclass classification task.

            Args:
                num_tokens: Integer specifying the number of tokens that should be the output of the final layer.
                input_dim: Dimension of the input embeddings.
                nhead: Integer specifying the number of heads for the `ConvBert` model.
                hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
                num_hidden_layers: Integer specifying the number of hidden layers for the `ConvBert` model.
                num_layers: Integer specifying the number of `ConvBert` layers.
                kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
                dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
        """

        self.model_type = "Transformer"

    def forward(self, last_hidden_state, attention_mask):
        hidden_inputs = self.convbert_forward(last_hidden_state, attention_mask)
        return hidden_inputs
