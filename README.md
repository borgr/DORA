# DORA The Explorer: Directed Outreaching Reinforcement Action-Selection  

This repository contains supplementary code for the ICLR2018 paper [DORA The Explorer: Directed Outreaching Reinforcement Action-Selection](https://openreview.net/forum?id=ry1arUgCW).

If you use any of the code related to this repository in a paper research etc., please cite:

```bibtex
@inproceedings{
    fox2018dora,
    title={{DORA} The Explorer: Directed Outreaching Reinforcement Action-Selection},
    author={Lior Fox and Leshem Choshen and Yonatan Loewenstein},
    booktitle={International Conference on Learning Representations},
    year={2018},
    url={https://openreview.net/forum?id=ry1arUgCW},
}
```

## Linear Approximation with tile-coding
Tile-coding features uses Richard's Sutton [implementation](http://www.incompleteideas.net/tiles/tiles3.html).
The experiments in the paper were performed using a version of the MountainCar environment with episode length of 1000 steps, following the definition in Sutton and Barto book. The gym version of MountainCar has by deafult an episode length of 200 steps. To register a 1000 steps version add the following snippet to the `gym/envs/__init__.py` file:
```python
register(
    id='MountainCarLong-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=1000,
    reward_threshold=-110.0,
)
```

To use the sparse reward version of the environment (reward=0 unless reaching goal) add the `--binreward` flag to execution.
Train a DORA agent (LLL-softmax) on the sparse-reward version of mountaincar (1000 episode steps):
~~~
$ python3 linapprox/main.py -a dora -e MountainCarLong-v0 --binreward
~~~

## DQN Implementation for ATARI
The implementation is based on the [atari-rl](https://github.com/brendanator/atari-rl) package, currently under [this fork](https://github.com/brendanator/atari-rl).
We added support for a two-streamed network for predicting Q and E values. Exploration bonus was added to the reward, and action-selection was e-greedy (more details can be found in section 4 of the paper).
To run the e-values based agent, add the `--e_network` flag to train the two-streamed network for predicting E-values, and the `--e_exploration_bonus` flag to use generalized counters (E-values) as exploration bonus.
~~~
$ python3 main.py --game freeway --e_network --e_exploration_bonus
~~~
