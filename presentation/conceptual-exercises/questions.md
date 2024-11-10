# Preliminaries 

For these questions, unless otherwise specified, we assume that $Q$ means the
$Q$ function associated with the optimal policy.

# Hypothetical RL situation

You are training an AI agent to play a simple game. It needs to navigate a grid
world that consists of a starting point (S), empty spaces, a goal square (G)
with a reward, and a trap square (T) with a negative reward. Each square on the
grid represents a state. The agent can move up, down, left, or right when
possible. The transition reward for reaching the goal is +10, for falling into
the trap is -10, and for any other step is -1 (to encourage the shortest path).
We'll call this last reward the "per-step reward." A discount factor $\gamma$ of
0.9 is applied.

The game ends immediately when the agent enters G or T.

Let's assume that our grid looks like the following:

```
S_T
___
__G
```

1. What is the immediate reward the agent receives when it is at `S` and moves
   one square down?
2. What is the Q-value of an optimal policy when the agent starts at `S` and
   moves one square down?
3. Again assuming we start at `S`, what is the path traced by one possible
   optimal policy before the game ends?
4. Let's assume that we try to play around with the per-step reward and
   discount factor. Is there any combination of per-step reward and discount
   factor that causes an agent with an optimal policy to never enter the goal
   square? Is there any combination of per-step reward and discount factor that
   causes an agent with a optimal policy to try to enter the trap square? 
5. Let's say that a genie tells you that the Q-function of the optimal policy
   given a position and a move is calculated by taking the Manhattan distance
   between the new position and G after making the move and then adding 10 to that
   value. The genie is wrong. Can you come up with a violation of Bellman's
   Equation to demonstrate to the genie that he's wrong?
