from torch import nn
import torch

from .code import PositionalEncoding

#  import torch_scatter
import torch.autograd.profiler as profiler
import util


class DepthOracleMemes(nn.Module):
    def __init__(self, latent, d_hidden, bins):

        self.latent = latent
        self.d_hidden = d_hidden
        self.bins = bins

        self.network = nn.Sequential(
            []
        )
    

    def forward(self, x):
        """
        x: tensor of shape (..., bins, dim)
        """


    def get_encodings(self, latent):
        """
        Here is where you take in latents of shape (SB, NS, B, K, latent)
        and then spit out whatever shape you will use in forward
        """


    @classmethod
    def from_conf(cls, conf, d_latent, **kwargs):
        # PyHocon construction
        return cls(
            d_latent,
            d_hidden=conf.get_int("d_hidden", 128),
            bins=conf.get_int("output_bins")
            **kwargs
        )




class DepthOracleNormals(nn.Module):
    def __init__(
        self,
        d_in,
        d_latent,
        bins=8,
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
            [nn.Linear(d_in + d_latent, d_hidden), nn.ReLU()] + \
            [nn.Linear(d_hidden, d_hidden), nn.ReLU()] * 7 + \
            [nn.Linear(d_hidden, bins), nn.Sigmoid()]
        )


    def forward(self, x):
        """
        :param x (..., d_latent + d_in)
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """

        with profiler.record_function("oracle_infer"):
            return self.network(x)


    def get_encodings(self, latent):
        """
        Here is where you take in latents of shape (SB, NS, B, K, latent)
        and then spit out whatever shape you will use in forward
        """


    @classmethod
    def from_conf(cls, conf, d_in, d_latent, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            d_latent,
            d_hidden=conf.get_int("d_hidden", 128),
            bins=conf.get_int("output_bins")
            **kwargs
        )
