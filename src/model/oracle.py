from torch import nn
import torch

from .code import PositionalEncoding

#  import torch_scatter
import torch.autograd.profiler as profiler
import util


class DepthOraclePool(nn.Module):
    def __init__(
        self,
        img_encoding_size,
        num_outputs = 8,
        num_layers = 8,
        hidden_size = 256,
        ray_encoding = "positional"
    ):
        """
        :param img_encoding_size The size of the PixelNeRF image encoding
        :param num_outputs Number of bins for our depth oracle's piecewise PDF
        """
        super().__init__()
        if ray_encoding == "positional":
            self.ray_encoding = PositionalEncoding()
        else:
            raise NotImplementedError("Oracle ray encoding not supported")
        # Need to encode both the origin and offset and another 2 for z_min and z_max 
        self.input_size = (self.ray_encoding.d_out * 2) + 2
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.oracle = nn.Sequential(
            [nn.Linear(self.input_size, self.hidden_size), nn.ReLU()] + \
            ([nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()] * 6) + \
            [nn.Linear(self.hidden_size, self.num_outputs)])


    def forward(self, rays):
        """
        :params rays Rays of shape (SB, B, 8) with [origin, offset, z_near, z_far]
            These rays are assumed to be in global coordinates
        Returns: A dictionary with the following elements:
            weights: shape (B, K)
            rgb_final: shape (B, 3)
            depth_final: shape (B) 
        """
        SB, B, _ = rays.shape

        with profiler.record_function("oracle_predict"):
            # Each batched example has a different pose, so do each independently
            for obj_idx in range(SB):
                obj_rays = rays[obj_idx]
                encodings = self.get_encodings(obj_rays, obj_idx)


    # def get_encodings(self, obj_rays, obj_idx, steps = 64):
    #     """
    #     This will return the concatenation of the max pool and average pool of all
    #         image encoded features along the path of the given rays

    #     We will do this by creating a linspace along each ray and then do the pooling
    #     over all the cells each point in the linspace overlaps with. 


    #     :param obj_rays Shape (B, 8) rays for a single image
    #     :param obj_idx Integer index into the superbatch dimension SB
    #     Returns: A tensor of encodings of shape (B, self.img_encoding_size * 2)
    #     """
    #     assert len(obj_rays) == 2
    #     B, _ = obj_rays.shape

    #     obj_ray_origins = obj_rays[:, :3].reshape(-1, 1, 3)
    #     obj_ray_directions = obj_rays[:, 3:6].reshape(-1, 1, 3)
    #     obj_ray_z_near = obj_rays[:, 6].reshape(-1, 1)
    #     obj_ray_z_far = obj_rays[:, 7].reshape(-1, 1)
    #     obj_ray_z_dist = obj_ray_z_far - obj_ray_z_near

    #     linspace = torch.linspace(0, 1, 64).reshape(1, -1)
    #     linspace = (linspace.to(obj_rays.device) * obj_ray_z_dist) + obj_ray_z_near

    #     assert linspace.shape == (B, steps)

    #     linspace = linspace.reshape(B, steps, 1)

    #     # We are going to get a queries tensor of shape B, steps, 3
    #     queries_xyz = (linspace * obj_ray_directions) + obj_ray_origins

    #     assert queries_xyz.shape == (B, steps, 3)

        
    def get_encodings2(self, model, rays, z_samp):
        """
        This will return the concatenation of the max pool and average pool of all
            image encoded features along the path of the given rays

        We will do this by creating a linspace along each ray and then do the pooling
        over all the cells each point in the linspace overlaps with. The linspace
        here is defined by z_samp which is the distances along each ray at which we
        will sample


        :param obj_rays Shape (B, 8) rays for a single image
        :param obj_idx Integer index into the superbatch dimension SB
        Returns: A tensor of encodings of shape (B, self.img_encoding_size * 2)
        """

        # shape (B, K, 3)
        points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
        
                
        


    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int("n_blocks", 5),
            d_hidden=conf.get_int("d_hidden", 128),
            beta=conf.get_float("beta", 0.0),
            combine_layer=conf.get_int("combine_layer", 1000),
            combine_type=conf.get_string("combine_type", "average"),  # average | max
            use_spade=conf.get_bool("use_spade", False),
            use_transformer=conf.get_bool("source_view_transformer", False),
            **kwargs
        )
