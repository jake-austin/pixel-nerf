from torch import nn
import torch

from .code import PositionalEncoding

#  import torch_scatter
import torch.autograd.profiler as profiler
import util


class DepthOracleNormals(nn.Module):
    def __init__(
        self,
        d_in,
        d_latent,
        d_out=8,
        d_hidden=128,
    ):
        """
        :param d_in input size
        :param d_out output size (number of bins)
        :param n_blocks number of Resnet blocks
        :param d_latent latent size
        :param d_hidden hiddent dimension throughout network
        :param use_transformer whether we use source view transformer or not
        """
        
        self.network = nn.Sequential(
            [nn.Linear(d_in + d_latent, d_hidden),
            nn.ReLU()] + \
            [nn.Linear(d_hidden, d_hidden), nn.ReLU()] * 7 + \
            [nn.Linear(d_hidden, d_out), nn.Sigmoid()]
        )


    def forward(self, x):
        """
        :param x (..., d_latent + d_in)
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """

        with profiler.record_function("oracle_infer"):
            return self.network(x)


    @classmethod
    def from_conf(cls, conf, d_in, d_latent, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            d_latent,
            d_hidden=conf.get_int("d_hidden", 128),
            d_out=conf.get_int("output_bins")
            **kwargs
        )
