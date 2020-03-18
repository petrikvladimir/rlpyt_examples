# Installation

```
first clone rlpyt from source
cr rlpyt
conda env create -f linux_cpu.yml
conda activate rlpyt

pip install -e .
pip install tensorboard
pip install future
pip install matplotlib
pip install PyQt5
```

# Optional library modifications

- in `rlpyt/utils/logging/logger.py` change `group_slash=False` to `group_slash=True`, it will group plots in tensorboard

# Notes

- The evaluation of the agent is not greedy (which is default in tf_agents).
In your model definition you can switch to greedy evaluation with (see `rl_example_2`):
```
For categorical dist.:
if self._mode == "eval":
    action = torch.argmax(agent_info.dist_info.prob, dim=-1)

For normal dist.:
if self._mode == "eval":
    action = agent_info.dist_info.mean
```

# Examples
## Example 1
Reacher 2D. The goal is to reach given goal position in 2D plane.

```
python rl_example_1.py # Train for 400 iterations 
tensorboard --logdir data # check localhost:6006 and see Return plot

python rl_example_1.py --test # evaluate using random policy
python rl_example_1.py --test --use_mode # evaluate using mean of the random policy
```

## Example 2
Reacher 2D with discrete action space.
```
python rl_example_1.py --greedy_eval
tensorboard --logdir data # check localhost:6006 and see Return plot

python rl_example_1.py --test
python rl_example_1.py --test --greedy_eval
```