if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import torch
import hydra
import dill
import time
from tqdm import tqdm
from gym import spaces
import collections
import numpy as np
import pymunk.pygame_util
from diffusion_policy.env.memory.memory_env_v4 import MemoryEnv_v4
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from typing import Dict, Sequence, Union, Optional
import cv2
from skvideo.io import vwrite

from diffusion_policy.policy.tedi_visualize_buffer import TEDiVisualizeBufferPolicy
from diffusion_policy.policy.diffusion_visualize_buffer import DiffusionVisualizeBufferPolicy
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy
import numpy as np

# Function to add the legend to each frame
def add_legend_to_frame(frame, legend, position=(12, int(768/2))):
    # Resize the legend to fit the frame without distorting aspect ratio
    h, w = frame.shape[:2]
    legend_aspect_ratio = legend.shape[1] / legend.shape[0]
    legend_width = int(w * 0.3)  # e.g., 10% of frame width
    legend_height = int(legend_width / legend_aspect_ratio)

    legend_scaled = cv2.resize(legend, (legend_width, legend_height))

    # Position the legend on the frame
    lx, ly = (position[0], int(h / 2 - legend_height / 2))
    sx, sy = lx + legend_scaled.shape[1], ly + legend_scaled.shape[0]

    if ly + legend_height > h or lx + legend_width > w:
        print("Legend dimensions exceed frame dimensions. Check the scaling and position.")
        return frame

    frame[ly:sy, lx:sx] = legend_scaled
    return frame


class MemoryEnvVisualizeBuffer(MemoryEnv_v4):
    def __init__(self,
                 legacy=False,
                 block_cog=None,
                 damping=None,
                 render_size=96,
                 reset_to_state=None,
                 render_action=True):
        # Call the parent constructor.
        super().__init__(legacy=legacy,
                         block_cog=block_cog,
                         damping=damping,
                         render_size=render_size,
                         reset_to_state=reset_to_state,
                         render_action=render_action)
        self.buffer = None

        self.color_options = {
            "carrot_orange": (55, 152, 240),  # BGR
            "robin_egg_blue": (205, 197, 13),
            "sgbus_green": (7, 233, 0),
            "slate_blue": (222, 83, 125),
            "penn_blue": (69, 16, 10)
        }
        self.current_color = "carrot_orange"  # Default color
    
    def set_path_color(self, color_name):
        if color_name in self.color_options:
            self.path_color = self.color_options[color_name]
        else:
            print(f"Color {color_name} not found. Using default color.")
            self.path_color = self.color_options["slate_blue"]

    def plot_path(self, img, path):
        if not path:
            return img

        path = np.array(path)
        path = (path / 512 * self.render_size).astype(np.int32)

        for i in range(1, len(path)):
            coord_start = tuple(path[i-1])
            coord_end = tuple(path[i])
            marker_size = int(2/96*self.render_size)
            
            overlay = img.copy()
            cv2.line(overlay, coord_start, coord_end, self.path_color, marker_size)
            img = cv2.addWeighted(overlay, 1, img, 0, 0)

        return img

    def set_buffer(self, buffer):
        self.buffer = buffer

    def set_buffer_diff_steps(self, buffer_diff_steps, diff_steps_max=99, diff_steps_min=-1):
        self.buffer_diff_steps = buffer_diff_steps
        self.diff_steps_max = diff_steps_max
        self.diff_steps_min = diff_steps_min

    def set_buffer_color(self, color_name):
        if color_name in self.color_options:
            self.current_color = color_name
        else:
            print(f"Color {color_name} not found. Using default color.")
    
    def _get_obs(self):
        return super()._get_obs()

    def draw_buffer(self, img, buffer):
        if buffer is None:
            return img
        
        n = buffer.shape[0]
        color = self.color_options[self.current_color]
        
        for i in range(n):
            coord = buffer[i]
            coord = (coord / 512 * self.render_size).astype(np.int32)
            marker_size = int(2/96*self.render_size)
            
            if img.shape[2] < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            overlay = img.copy()
            cv2.circle(overlay, coord, marker_size, color, -1)
            img = cv2.addWeighted(overlay, 1, img, 0, 0)
            if img is None: 
                print("img is None after cv2")

        return img

    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        img = self.draw_buffer(img, self.buffer)
        return img


if __name__ == "__main__":
    
    # 1. Load policy
    checkpoint = "data/outputs/2025.03.12/16.53.13_train_tedi_ddim_unet_lowdim_pusht_memory_lowdim/checkpoints/epoch=0350-test_mean_score=0.494.ckpt"
    #checkpoint = "data/outputs/2025.03.12/13.22.36_train_diffusion_transformer_lowdim_pusht_memory_lowdim/checkpoints/epoch=0800-test_mean_score=0.616.ckpt"

    vis_policy = TEDiVisualizeBufferPolicy(checkpoint)
    #vis_policy = DiffusionVisualizeBufferPolicy(checkpoint)
    device = torch.device("cuda:0")
    vis_policy.to(device)
    vis_policy.eval()
    obs_horizon = vis_policy.n_obs_steps

    # limit enviornment interaction to 200 steps before termination
    max_steps = 50
    env = MemoryEnvVisualizeBuffer(render_size=768)
    env.set_buffer_color("robin_egg_blue")
    env.set_path_color("slate_blue")
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(100000)

    # get first observation
    obs = env.reset()

    # Create the legend (you might need to adjust these based on the expected range)
    #legend_image = env.create_legend(0, 99)  # Update min and max values based on your application

    #legend_image = cv2.imread('legend.png', cv2.IMREAD_COLOR)

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = []
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="EvalMemoryEnv") as pbar:
        while not done:
            marker_path = [env.agent.position]
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_dict = {
                "obs": torch.from_numpy(np.stack(obs_deque, axis=0)).to(device, dtype=torch.float32).unsqueeze(0),
            }
            action, img_frames = vis_policy.predict_action(obs_dict, env)
            imgs.extend(img_frames)
            buffer = action['action_pred'].detach().to('cpu').numpy()[0]
            env.buffer = buffer

            action = action['action'].detach().to('cpu').numpy()[0]
            #Sleep a tiny bit so that we can see the prediciton
            #time.sleep(0.1)

            # Before moving, plot the current plan
            # Remove first (obs) action from buffer
            env.buffer = env.buffer[1:]
            imgs.append(env.render(mode='rgb_array'))

            

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                
                ## Render
                # Remove the leftmost action from the env buffer
                marker_path.append(env.agent.position)
                env.buffer = env.buffer[1:]
                frame = env.render(mode='rgb_array')
                frame = env.draw_buffer(frame, np.array(marker_path))
                imgs.append(frame)

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break
            
            # Plot the path
            img = env.render(mode='rgb_array')
            img = env.draw_buffer(img, np.array(marker_path))
            imgs.append(img)
            print(f"Len of marker path: {len(marker_path)}")

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    from IPython.display import Video
    video_path = 'visualization/video/vis_tedi.mp4'
    vwrite(video_path, imgs)
    print('Done saving to ', video_path)

    # Save the 2nd frame as an image
    # img_path = 'visualization/image/vis_tedi.png'
    # cv2.imwrite(img_path, imgs[7])
    # print('Done saving to ', img_path)