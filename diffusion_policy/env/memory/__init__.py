from gym.envs.registration import register
import diffusion_policy.env.memory

register(
    id='memory-keypoints-v0',
    entry_point='envs.memory.memory_keypoints_env:MemoryKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0
)