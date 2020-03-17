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

- it seems that evaluation is random (in tf_agents it was greedy policy instead); can be modified based on mode in gaussian.py

# Examples
## Example 1
Reacher 2D. The goal is to reach given goal position in 2D plane.

```
python rl_example_1.py # Train for 400 iterations 
tensorboard --logdir data # check localhost:6006 and see Return plot

python rl_example_1.py --test # evaluate using random policy
python rl_example_1.py --test --use_mode # evaluate using mean of the random policy
```
