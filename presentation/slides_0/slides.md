% Reinforcement Learning
% Changlin Li (mail@changlinli.com)

# What is Reinforcement Learning

+ An agent that can be in any one of a variety of states that must perform
  actions in an environment
+ Train using short-term rewards based on agent state and behavior to guide
  long-term behavior
+ Human designer does not know the optimal long-term strategy

# How does it differ from standard supervised/unsupervise learning

+ Supervised learning requires labeling with known good behavior
    * In an RL context humans may not know what the optimal strategy is and so
      can't generate optimal labels
+ Unsupervised learning is generally descriptive
    * Way of visualizing/understanding data, e.g. clustering
    * Doesn't come up with strategies on its own
    * Want an agent that is capable of probing

# Example

+ Game of chess
+ Humans don't know what the optimal long-term straegy is
+ But we have a good idea of what immediate rewards should look like
    * E.g. if you checkmate your opponent on the next move you should get very
      high reward for that move
    * Likewise if you can get checkmated on the next move by playing a certain
      move you should get very low reward for that move
+ Can use intermediate rewards such as taking pieces

# Simpler example 

+ Mazes
+ RL is a bit of overkill here because humans do know what the optimal 

# Basic RL strategy is Q-learning

+ We provide R (short-term reward function)
+ Q is the optimal reward function, i.e. it is the maximal possible long-term
  reward for performing a certain action
+ Q can be defined recursively
    * `Q(s, a) = R(s, a) + max_a_i(Q(s', a_i))
    * Known as Bellman's Equation

# Example R and Q in maze

+ State is a coordinate in the maze
+ Actions are up, down, right, left
+ R(s, a) = 1000 if action a gets you to the exit on your next move, zero otherwise

# Example R and Q in maze

We are at (1, 1) in a maze and we want to calculate Q for the action UP

+ R((1, 1), UP) = 0 (assuming exit is not at (1, 2))
+ So Q((1, 1), UP) = 0 + max(Q((1, 2), UP), Q((1, 2), DOWN), Q((1, 2), LEFT), Q((1, 2), RIGHT))

# This example is a bit trivial

+ Since R is always 0 or 1000 and you can always eventually get to the exit from
  anywhere in the maze, Q(s, a) is always 1000

# This is bad for agent training

+ The agent will probably learn to go in circles, because no incentive to go to
  exit quickly

# Add small negative reward (penalty) for every move

+ R(s, a) = 1000 if a gets you to the exit on the next move
+ Otherwise -1

# Let's go to code!
