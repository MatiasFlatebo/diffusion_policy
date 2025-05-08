import numpy as np
import click
import collections
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht_memory_state.pusht_memory_keypoints_env_state import PushTMemoryKeypointsEnv
import pygame

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
def main(output, render_size, control_hz):
    """
    Collect demonstration for the Push-T task.
    
    Usage: python demo_pusht.py -o data/pusht_demo.zarr
    
    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area. 
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """
    
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # create PushT env with keypoints
    kp_kwargs = PushTMemoryKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTMemoryKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()
    
    # New! 
    past_action_horizon = 4  # Must match the configuration
    action_dim = 2  # Same as the action space dimension

    def reset_past_actions():
        """ Helper function to reset the past actions buffer. """
        return collections.deque([np.zeros(action_dim) for _ in range(past_action_horizon)], maxlen=past_action_horizon)
    
    # episode-level while loop
    while True:
        past_actions = reset_past_actions()
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        print(f'starting seed {seed}')

        # set seed for env
        env.seed(seed)
        
        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='human')
        
        while True:
            mouse_position = pygame.mouse.get_pos()
            agent_position = env.agent.position
            distance = np.linalg.norm(np.array(mouse_position) - np.array(agent_position))
            if distance < 30:
                break
            pygame.event.pump()
        
        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        pygame.display.set_caption(f'plan_idx:{plan_idx}')
        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry=True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # handle control flow
            if retry:
                # New! Reset past action buffer before retry
                past_actions = reset_past_actions()
                break
            if pause:
                continue
            
            # get action from mouse
            # None if mouse is not close to the agent
            act = agent.act(obs)
            if not act is None:
                #New!
                past_actions.append(act)

                # teleop started
                # state dim 2+3
                goal_index = float(info['goal_pose_idx'])
                state = np.concatenate([info['pos_agent'], info['block_pose'], [goal_index]])
                # discard unused information such as visibility mask and agent pos
                # for compatibility

                # Note: Last element of obs is goal index, not part of keypoints
                # total obs size = keypoints + mask + 1 extra goal index
                obs_no_goal = obs[:-1]  # remove goal index
                half = obs_no_goal.shape[0] // 2

                # Split keypoints and mask
                keypoints_flat = obs_no_goal[:half]
                keypoint = keypoints_flat.reshape(-1, 2)[:9]

                # New!
                past_actions_flat = np.array(past_actions).flatten()

                data = {
                    'img': img,
                    'state': np.float32(state),
                    'keypoint': np.float32(keypoint),
                    'action': np.float32(act),
                    'past_actions': np.float32(past_actions_flat),
                    'n_contacts': np.float32([info['n_contacts']])
                }
                episode.append(data)
                
            # step env and render
            obs, reward, done, info = env.step(act)
            img = env.render(mode='human')
            
            # regulate control frequency
            clock.tick(control_hz)
        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')
        else:
            print(f'retry seed {seed}')


if __name__ == "__main__":
    main()
