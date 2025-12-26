import sys

sys.path.insert(0, ".")
import gymnasium as gym

# Create the environment
env = gym.make("Blokus-v0")

# Reset the environment
state = env.reset()

# Render the initial state
env.render()

# Take a random action
action = env.action_space.sample()

# Execute the action
state, reward, done, truncated, info = env.step(action)

# Render the new state
env.render()

# Close the environment
env.close()
