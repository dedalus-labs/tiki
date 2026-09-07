# Copyright © 2023 Apple Inc.

from tiki.nn.layers.activations import (
    CELU,
    ELU,
    GELU,
    GLU,
    SELU,
    HardShrink,
    Hardswish,
    HardTanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    PReLU,
    ReLU,
    ReLU2,
    ReLU6,
    Sigmoid,
    SiLU,
    Softmax,
    Softmin,
    Softplus,
    Softshrink,
    Softsign,
    Step,
    Tanh,
    celu,
    elu,
    gelu,
    gelu_approx,
    gelu_fast_approx,
    glu,
    hard_shrink,
    hard_tanh,
    hardswish,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    mish,
    prelu,
    relu,
    relu2,
    relu6,
    selu,
    sigmoid,
    silu,
    softmax,
    softmin,
    softplus,
    softshrink,
    softsign,
    step,
    tanh,
)
from tiki.nn.layers.base import Module
from tiki.nn.layers.containers import Sequential
from tiki.nn.layers.convolution import Conv1d, Conv2d, Conv3d
from tiki.nn.layers.convolution_transpose import (
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from tiki.nn.layers.distributed import (
    AllToShardedLinear,
    FullyShardedModule,
    QuantizedAllToShardedLinear,
    QuantizedShardedToAllLinear,
    ShardedToAllLinear,
    fully_shard,
)
from tiki.nn.layers.dropout import Dropout, Dropout2d, Dropout3d
from tiki.nn.layers.embedding import Embedding
from tiki.nn.layers.linear import Bilinear, Identity, Linear
from tiki.nn.layers.normalization import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    RMSNorm,
)
from tiki.nn.layers.pooling import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)
from tiki.nn.layers.positional_encoding import ALiBi, RoPE, SinusoidalPositionalEncoding
from tiki.nn.layers.quantized import (
    QQLinear,
    QuantizedEmbedding,
    QuantizedLinear,
    quantize,
)
from tiki.nn.layers.recurrent import GRU, LSTM, RNN
from tiki.nn.layers.transformer import (
    MultiHeadAttention,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from tiki.nn.layers.upsample import Upsample
