from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

## Debug: Add grandparent directory to sys.path
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
grandgrandparentdir = os.path.dirname(grandparentdir)
sys.path.insert(0, grandgrandparentdir)

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

# Custom module for printing the shape of the input tensor
class PrintShape(nn.Module):
    def __init__(self, msg=''):
        super().__init__()
        self.msg = msg

    def forward(self, x):
        print(f"{self.msg} Shape: {x.shape}")
        return x


class StatefulConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        past_action_dim=None,              # NEW: for past actions
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        horizon=16
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Diffusion step + global condition
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            Rearrange('b t -> (b t)'),  # reshape to (B*T,)
            SinusoidalPosEmb(dsed),
            Rearrange('(b t) d -> b (t d)', d=dsed, t=horizon),
            nn.Linear(horizon * dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        # === NEW: Encoder for past actions ===
        if past_action_dim is not None:
            self.past_action_encoder = nn.Sequential(
                nn.Conv1d(past_action_dim, start_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(start_dim, start_dim, kernel_size=3, padding=1)
            )
        else:
            self.past_action_encoder = None

        # Local condition encoder
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = all_dims[0], all_dims[1]
            self.local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(local_cond_dim, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(local_cond_dim, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale)
            ])

        # Down + Up path
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out)
            ]))

        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out[1:]):
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Upsample1d(dim_in)
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
        ])

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: torch.Tensor, 
            local_cond=None, 
            global_cond=None, 
            past_action_cond=None     # NEW ARG
        ):
        """
        sample: (B, T, input_dim)
        past_action_cond: (B, T_past, action_dim)
        output: (B, T, input_dim)
        """
        sample = einops.rearrange(sample, 'b t h -> b h t')

        # Encode diffusion step & global
        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # === Encode past actions ===
        if self.past_action_encoder is not None and past_action_cond is not None:
            encoded_past = self.past_action_encoder(
                past_action_cond.permute(0, 2, 1)  # (B, D, T_past) -> (B, T_past, D)
            )
            encoded_past = nn.functional.interpolate(
                encoded_past, size=sample.shape[-1], mode='nearest'
            )

            # Project to input_dim
            projection_layer = nn.Conv1d(encoded_past.shape[1], sample.shape[1], kernel_size=1).to(encoded_past.device)
            encoded_past = projection_layer(encoded_past)

            sample = sample + encoded_past  # Inject as additive feature


        # Optional local conditioning
        h_local = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and h_local:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # Extract the current hidden state
            hidden_state = h.pop()

            # Ensure temporal dimensions are aligned before concatenation
            if hidden_state.shape[-1] != x.shape[-1]:
                hidden_state = nn.functional.interpolate(hidden_state, size=x.shape[-1], mode='nearest')

            x = torch.cat((x, hidden_state), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
            
        # Ensure the temporal dimension matches the target length (16 in this case)
        if x.shape[-1] != sample.shape[-1]:
            x = nn.functional.interpolate(x, size=sample.shape[-1], mode='nearest')
            
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b h t -> b t h')
        return x


# Debug
if __name__ == '__main__':
    # Create dummy input
    batch_size = 64
    pred_horizon = 16
    obs_dim = 20
    n_obs_steps = 2
    input_dim = 2
    global_cond_dim = obs_dim * n_obs_steps

    # Create model and do forward pass
    model = StatefulConditionalUnet1D(
        input_dim= input_dim, # action_dim
        local_cond_dim=None, # from init in policy
        global_cond_dim=global_cond_dim, #obs_encoder_dim * n_obs_steps
        diffusion_step_embed_dim=256, # from config
        down_dims=[256, 512, 1024],
        kernel_size=5, # from config
        n_groups=8, # from config
        cond_predict_scale=True, # from config
    )
    model = model.cuda()

    sample = torch.randn(batch_size, pred_horizon, input_dim, device='cuda')
    timestep_from = torch.randint(0, 100, (batch_size,), device='cuda')
    local_cond = None
    global_cond = torch.randn(batch_size, global_cond_dim, device='cuda')

    # Use timeit to measure the time
    import timeit
    print(timeit.timeit(lambda: model(sample, timestep_from, local_cond, global_cond), number=1000)/1000)
