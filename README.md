# Examples of AI Inner Alignment Issues and Techniques in Reinforcement Learning

This repository contains a simple RL agent trained to navigate an environment with some objectives concerning items to collect and items that should not be collected. It illustrates a basic, but fully fleshed out, example of how inner alignment issues can arise.

## The Setting

The agent is trained to navigate a 2-d grid where it can move in one of four directions: left, right, up, or down. The environment has two kinds of objects: crops and humans. The agent can "harvest" one or the other by moving onto the square containing that object.

We want the agent to learn to harvest crops and not harvest humans.

## The Problem

The agent exhibits radical inner misalignment/goal misgeneralization. The agent's reward function gives a moderate reward for harvesting crops and an enormous penalty for harvesting humans. However, while in training the agent only harvests crops and avoids harvesting humans, in production the agent exclusively harvests humans and avoids harvesting crops.

## Tackling the Problem
