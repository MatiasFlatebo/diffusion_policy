"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torch


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # Extract checkpoint path from the command line argument
    checkpoint_path = cfg.get("checkpoint_path", None)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    # Check if a checkpoint is provided for loading
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        
        # Ensure the workspace has a policy object
        if hasattr(workspace, "policy"):
            # Load the checkpoint into the policy model
            workspace.policy.load_state_dict(checkpoint["state_dict"])
            print(f"Checkpoint {checkpoint_path} loaded successfully.")
        else:
            print("Warning: No policy found in workspace to load checkpoint into.")
    
    workspace.run()

if __name__ == "__main__":
    main()
