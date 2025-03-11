from gym.envs.registration import register
import diffusion_policy.env.pusht_memory

register(
    id='pusht-memory-keypoints-v0',
    entry_point='envs.pusht_memory.pusht_memory_keypoints_env:PushTMemoryKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0
)