# Reversi Gym Environment

This repository contains a custom implementation of the Reversi game environment using the `gymnasium` library. It includes the environment setup and an example of how to interact with it.

## Files

- `gym_reversi.ipynb`: A Jupyter notebook demonstrating how to set up and run the Reversi environment.
- `reversi_random_template.py`: The Python script that defines the custom Reversi environment.

## Installation

To install the required dependencies, run:

```bash
pip install gymnasium numpy
```

## Usage

### Running the Jupyter Notebook

The `gym_reversi.ipynb` notebook provides an example of how to initialize and interact with the Reversi environment. To run the notebook:

1. Open the notebook in Jupyter.
2. Execute the cells to see the environment in action.

### Reversi Environment Details

The `ReversiEnv` class in `reversi_random_template.py` includes the following methods:

- **`__init__(self, render_mode=None, size=8)`**: Initializes the environment with a board of specified size.
- **`possible_moves(self, player, state)`**: Returns a list of possible moves for a player given the current state.
- **`flip_piece(self, state, action, player)`**: Flips the pieces on the board based on the action taken.
- **`scoring(self, state, player)`**: Calculates the score for a player.
- **`is_over(self)`**: Checks if the game is over.
- **`reset(self)`**: Resets the game to the initial state.
- **`step(self, action)`**: Executes an action and returns the next state, reward, termination status, and additional info.
- **`render(self)`**: Renders the current state of the board.

### Example Code

Here's a snippet to get you started with the Reversi environment:

```python
import gymnasium as gym
from reversi_random_template import ReversiEnv

# Register the environment
gym.envs.registration.register(
    id='Reversi-v0',
    entry_point='reversi_random_template:ReversiEnv',
)

# Create the environment
env = gym.make('Reversi-v0', render_mode='text')

# Reset the environment
obs, info = env.reset()

# Render the initial state
env.render()

# Example of taking a step
action = (2, 3)  # Replace with your action
obs, reward, done, truncated, info = env.step(action)

# Render the state after the action
env.render()
```


## Result

<video width="600" controls>
  <source src="Video/sample.mp4" type="video/mp4">
</video>

