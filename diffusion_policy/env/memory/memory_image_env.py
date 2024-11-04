from gym import spaces
from diffusion_policy.env.memory.memory_env import MemoryEnv
import numpy as np
import cv2

class MemoryImageEnv(MemoryEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            damping=None,
            render_size=96):
        super().__init__(
            legacy=legacy, 
            damping=damping,
            render_size=render_size,
            render_action=False)
        
        self.memory_goal = None
        ws = self.window_size
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
    
        self.render_cache = None
    
    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }
        
        if self.memory_goal is not None:
            goal_coord = (self.memory_goal / 512 * 96).astype(np.int32)
            cv2.drawMarker(img, goal_coord,
                        color=(0, 255, 0), markerType=cv2.MARKER_STAR,
                        markerSize=8, thickness=2)
        
        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
            
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache
