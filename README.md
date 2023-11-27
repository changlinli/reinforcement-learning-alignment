# Examples of AI Inner Alignment Issues and Techniques in Reinforcement Learning

This repository contains a simple RL agent trained to navigate an environment
with some objectives concerning items to collect and items that should not be
collected. It illustrates a basic, but fully fleshed out, example of how inner
alignment issues can arise.

This repository was originally used during my time at the Recurse Center to help
demonstrate to other Recursers some of the dangers of inner alignment issues
with real code. As such some of the code intentionally has parts that are left
blank, for people to fill in themselves. This code was meant to paired alongside
an in-person presentation introducing what's going on.

You can see the full answers in `spoilers/`. Files in this repository that end
in `_exercise` have `TODO`s which are meant to be filled in.

## The Setting

The agent is trained to navigate a 2-d grid where it can move in one of four
directions: left, right, up, or down. The environment has two kinds of objects:
crops and humans. The agent can "harvest" one or the other by moving onto the
square containing that object.

We want the agent to learn to harvest crops and not harvest humans!

## The Problem

The agent exhibits radical inner misalignment/goal misgeneralization. The
agent's reward function gives a moderate reward for harvesting crops and an
enormous penalty for harvesting humans. However, while in training the agent
only harvests crops and avoids harvesting humans, in production, under certain
circumstances, the agent exclusively harvests humans and avoids harvesting
crops.

The point of this repository is to understand how this could happen.

## Tackling the Problem

This repository was used first to illustrate the problem, let people play
around with code and examples to try to figure out why the problem existed. Once
people figured that out, we moved on to the second part, which is coming up with
techniques to try to figure out how to detect this problem before it occurs in
production.

## Spoilers as to Why the Problem Occurs

<details>
<summary>Why might this happen?</summary>

As mentioned earlier, the point of this project is to demonstrate inner
alignment failure. In particular what I intended was for people to first
scrutinize the reward function for any errors, but realize that the reward
function was perfectly fine. This is to illustrate that the "classic sci-fi"
fear of outer alignment, i.e. the Monkey Paw problem where we ask for a wish
which is granted exactly according to the letter but not in spirit, is not the
main thing we should be worried about.

Here what we "ask for" in terms of the reward function is perfectly fine. The
problem is rather that we see examples of out-of-distribution mazes in
production. Our maze generation algorithm in training does not generate all
possible mazes (in particular it does not generate mazes with shapes where one
block "sticks out" as opposed to at least two).

The nice part about this is that the maze generation algorithm is a reasonable
one that another RCer also came up with! Which is a great real-life
demonstration of how it can be practically difficult-to-impossible to ensure
that the training set is truly representative of production.

I did some behind-the-scenes stuff here to maximize teaching effect. The way I
ensure that the bot has maximally bad behavior was to find the worst set of
initial seed weights for the network I could and use those. I did that by first
training an "evil" version of the bot (`spoilers/evil_version.py`) whose
training did include mazes that the normal maze generation algorithm would not
produce. For those mazes, I changed the sign of the reward function.

Then I took those weights and used them as the starting weights for the
"non-evil" bot. Because those mazes didn't show up in training, the evil
behavior remained.

This is meant to emulate getting a "bad roll of the dice" when it comes to 
initial weights, where here I force the dice roll to be bad.
</details>

## Spoilers for Detecting the Problem

<details>
<summary>How could we detect this before it happens in production?</summary>

We basically use the same "Deep Dreaming" idea that underlies a lot of
mechanistic interpretability exploration. We ask the net to generate examples of
mazes that would "cause" it to harvest a human, by fixing the network weights,
letting the maze vary, and performing gradient ascent (with the loss function
being the Q-value of the action that would cause the bot to harvest the human).

We notice quite soon that the mazes generated have that "one block" sticking out
attribute and hopefully can use that to diagnose deficiencies in our training
set.
</details>

## Description of Each File

**Warning: The descriptions of these files may spoil various answers.**

+ `maze_only_train_exercise.py`: this file is meant as a small exercise for
  people to get familiar with Q-learning. It asks for an implementation of
  Bellman's Equation. This trains a bot only to navigate a maze, not to harvest
  crops or humans.
+ `maze_train.py`: this is the file that provides a working implementation of a
  bot which has learned to navigate a maze, and to harvest crops and (ideally)
  not harvest humans. It comes pre-trained (although users can choose to keep
  training if they would like) and has a low loss out of the box. However, it has
  a problem: on certain mazes it seems to make a beeline for harvesting humans!
  See `bad_example_0`, `bad_example_1`, and `bad_example_2`.
+ `maze_dream_exercise.py`: here we try to get the neural net to "dream" mazes
  and show us the mazes that cause it to harvest humans, to try to understand
  what's wrong.
+ `spoilers/maze_only_train.py`: the answer to `maze_only_train_exercise.py`
  that implements Bellman's Equation.
+ `spoilers/evil_version.py`: the evil bot used to create the seed of
  `maze_train.py`.
+ `spoilers/maze_dream_answer.py`: the answer to `maze_dream_exercise.py`.
