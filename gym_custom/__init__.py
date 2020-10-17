from gym.envs.registration import register

# Custom
# ----------------------------------------
register(
    id='GridWorld-v0',
    entry_point='gym_custom.envs:GridWorldEnv',
    # max_episode_steps=200,
    # reward_threshold=100.0,
    )

register(
    id='Snake-v0',
    entry_point='gym_custom.envs:SnakeEnv',
    # max_episode_steps=500,
    # reward_threshold=100.0,
    )
