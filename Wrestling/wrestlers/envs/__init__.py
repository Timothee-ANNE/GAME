# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from envs.wrestling_env import WrestlingEnvClass
from gymnasium.envs.registration import register

register(
    id = 'WrestlingEnv-v0',
    entry_point = 'envs.wrestling_env:WrestlingEnvClass',
    max_episode_steps = 500
)


# %%
