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



We will train an agent using RL techniques via pen and paper to get a better
intuition for what role Bellman's equation plays in Q-learning.

To make this feasible, we will start with an extremely simple game. We have a
board with three cells laid side-by-side. They are respectively cells `A`, `B`,
and `C`. The agent starts at a random cell at the beginning of the game and can
take turns deciding whether to go left or right. No matter where you start, the
objective of the game is to move to the right side of the board.

This results in the following outcomes:

+ If the agent is at cell `A`:
    * Going left does nothing, the agent remains on cell `A` at the beginning of
      the next turn and receives a reward of -0.1
    * Going right moves the agent to cell `B` and the agent receives a reward of
      -0.1
+ If the agent is at cell `B`:
    * Going left moves the agent to cell `A` and the agent receives a reward of
      -0.1
    * Going right moves the agent to cell `C` and the agent receives a reward of
      -0.1
+ If the agent is at cell `C`:
    * Going left moves the agent to cell `B` and the agent receives a reward of
      -0.1
    * Going right ends the game and the agent receives a reward of 1

Note that most moves (except the final move from cell `C` that ends the game)
have a mild negative reward penalty to encourage the agent to find the solution
with the fewest moves.

An ASCII representation of the board is given below.

```
-------
|A|B|C|
-------
```

We have two additional rules. There is a discount factor ($\gamma$) of 0.9.
Also, the game always ends after 5 turns no matter which cell the agent is
on.

*Exercise*

> Let's say our policy $\pi$ is to always go left, no matter what square we are
> at. If our current cell is cell $B$, what is the immediate reward I get after
> following this policy (i.e. what is the reward I get for performing one move)?
> What is $Q_\pi(B, right)$ where $Q_\pi(s, a)$ is the $Q$-value, i.e. total
> return, if every move performed after action $a$ is done according to the
> policy $\pi$? What is $Q_\pi(B, left)$? What is $Q_\pi(C, right)$?

<details>
<summary>Solution</summary>
The immediate reward at $B$ when following $\pi$ is -0.1 (following $\pi$ means
we go one step to the left and end up at $A$).

$Q_\pi(B, right) = -0.1 + (-0.1 \cdot 0.9) + (-0.1 \cdot 0.9 ^2) + (-0.1 * 0.9 ^3) + (-0.1 * 0.9 ^4) = -0.40951$, where the agent goes from B to C and then continues to follow its policy of going left and goes back to B and then to A and is stuck there for two more turns before the game hits the five turn maximum and ends.

$Q_\pi(C, right) = 1$, where the agent immediately ends the game by going right.

---
</details>

*Exercise*

> Based on this description and reward function, what is the optimal policy for
> an agent to follow? I.e. what strategy should the agent use to maximize the
> total reward it receives for any given game?

<details>
<summary>Solution</summary>
It should just always go to the right. It should never go left.
---
</details>

We're going to train this agent using Q-learning, but simply using pen and
paper. As a reminder, a $Q$ function takes in a state and an action and outputs
some number. In this case, we'll let our state be which cell the agent is
currently at, and our action will be either $left$ (i.e. move left) or $right$ (i.e.
move right).

We only have a total of six possible state, action pairs, so we can represent
our $Q$ function as a six element lookup table. Initialize this lookup table to
whatever you want. The below is one example of such a random initialization (but
again you can use whatever values you want).

```
A, right -> -1
A, left -> -3
B, right -> 4
B, left -> 0
C, right -> -1
C, left -> 2
```

*Exercise*:

> This game is simply enough that it is feasible to solve what the optimal $Q$
> function is directly without resorting to Q-learning via Bellman's equation to
> find what it is. What is the optimal $Q$ function? That is what is the $Q$
> value of each of the six possible state-action pairs if the agent is looking
> to maximize total reward at the end of a game?
>
> We can then compare what we get after running Q-learning to what we know the
> optimal $Q$ function should be.

<details>
<summary>Solution</summary>
```
A, right -> 0.62
A, left -> 0.458
B, right -> 0.8
B, left -> 0.458
C, right -> 1.0
C, left -> 0.62
```
---
</details>

Now use this $Q$ function to determine what the agent's policy should be.

Since this is the optimal $Q$ function, the optimal policy of the agent should
just be to do whatever action has the highest $Q$ value. You should find that
this means no matter which cell the agent is on, the agent should move right.

We're now going to use Bellman's equation to iteratively update our $Q$ function
and train our $Q$ function to go from random initial values to the correct
values of the optimal $Q$ function. That means we're going to take a lot of
state-action pairs, calculate the left-hand side and right-hand side of the
optimal Bellman equation, and then force them to be equal if they are not
already equal.

More concretely, do the following steps:

1. Choose a random state $s$ (one of $A$, $B$, or $C$) and a random action $a$
   (one of either $left$ or $right$)
2. Calculate both sides of Bellman's equation based on your current $Q$
   function. That is calculate $Q(s, a)$ and $R + \gamma \text{max} _{a'}Q(s',
   a')$. For example if our $s$ and $a$ were $B$ and $left$ and we use the
   random initialization described above, then $Q(s, a) = 0$ and $\text{max}
   _{s', a'}Q(s', a') = -1$, since $s'$ is the state after making our move, i.e.
   cell $A$ which means we are looking for the max among $Q(A, left)$ and $Q(A,
   right)$, which are $-3$ and $-1$ respectively. This means that right-hand
   side of Bellman's equation after taking into account $R$ and $\gamma$ is
   $-0.1 + 0.9 \cdot -1 = -1$
3. If the two sides of Bellman's equation are the same, do nothing. Otherwise we
   want to force the two sides of Bellman's equation to equal each other. To do
   this, we'll do the most brute-force method: we'll just update our $Q$
   function so that for our chosen $s$ and $a$, $Q(s, a)$ is now the value of
   our calculated value for the right-hand of Bellman's equation.

*Exercise*:

> Actually carry out all the above steps. Keep doing this until the $Q$ function
> stops changing on matter what state-action pair you use. You can do this in
> what order you want (just make sure that you end up iterating through every
> possible state-action pair, potentially multiple times). The exact number of
> times you will need to do this depends on the order you choose, but usually
> this will take about 6-12 iterations.

You should find that no matter what starting values you chose for your $Q$
function, you end up with the values of the optimal $Q$ function you found
earlier.

As a reminder, what we've done is used Bellman's equation essentially as a
consistency check, and have learned the optimal $Q$ function by simply checking
whether our $Q$ function is consistent according to Bellman's equation (i.e.
whether the two sides of Bellman's equation equal each other) and updating our
$Q$ to force it to be consistent if not.

Deep Q-Learning (i.e. DQN) follows the exact same general steps, but with neural
nets instead of a lookup table. So we'll start with a neural net with randomized
parameters that takes in a state-action pair and spits out a $Q$ value. We'll
calculate both sides of Bellman's equation as normal, but this time, to force
the two sides of Bellman's equation to be consistent, we'll take their
difference as a loss, and run gradient descent on the neural net's parameters to
reduce the loss. Then we repeat this process for new state-action pairs up to
some stopping point
