# Preliminaries

Let's explore some basic RL concepts using a simple turn-based single player
game. There is a board with five squares, laid out in four corners as shown
below.

```
  A
B E C
  D
```

Each corner has an item. That is squares A, B, C, and D each begin the game with
an item on that square. We'll name the items after the square that it is on. So
we have items A, B, C, and D.

An agent starts at square E. It can (and must) make a move of left, right, up,
or down. As soon as it makes a move, it collects the item at the square it moves
to, if an item exists. At the end of every turn, the agent is returned to square
E.

Once an item has been collected, it disappears forever. So for example if an
agent goes to square A twice, it collects item A the first time and then
collects nothing the second the time.

Every time an agent collects an item it receives a reward as follows:

+ When it collects item A it gets a reward of 10
+ When it collects item B it gets a reward of 1
+ When it collects item C it gets a reward of 5
+ When it collects item D it gets a reward of -10

The game ends when item C is collected. Otherwise the game runs for a maximum of
10 turns. At the end of ten turns, even if item C has not been collected, the
game ends.

Every full run of the game is called an *episode*. An episode is often
represented as a list of triples $(s, a, r)$ denoting a given state, the action
that was taken from that state, and the reward the agent received from that
action. In our case, an episode can never be longer than 10 triples. A set of
contiguous triples (i.e. that resulted from a contiguous set of actions) is
called a *trajectory*.

The strategy that an agent uses to determine what move it should make from a
given state of the game is called the agent's *policy* and is often denoted by
$\pi$.

The total amount of reward the agent accrues over an episode is called the
*return*. Return is also sometimes used to refer to the total reward an agent
accrues after starting from a state part-way through a game. The maximum total
reward an agent can accrue from state $s$ after taking action $a$ is called the
$Q$-value of $s$ and $a$ and is denoted $Q(s, a)$. This means that in general
$Q$ is a function from state-action pairs to real numbers.

The policy that gives the highest return is called the optimal policy and
reinforcement learning is all about finding the optimal policy.

We sometimes also talk about the total reward an agent can accrue from some
state $s$ after making an action $a$ if it follows a policy $\pi$ where $\pi$
may be suboptimal. In that case we often subscript $Q$ with $Q_\pi$ where
$Q_\pi(s, a)$ denotes the total return when we follow policy $\pi$ after having
performed action $a$ in state $s$. Note that to calculate this $Q$ value we only
follow $\pi$ after performing $a$. $\pi$ may not have chosen to perform action
$a$ from state $s$ if it was left to its own devices.

*Exercise*:

> Let's say we use the policy that goes left, then up, then left, then up,
> etc. until the game ends after 10 turns. What is the return of a game if we
> play according to this policy? Remember at the end of every move the agent is
> returned to square E.

*Exercise*:

> Let's say we start from the state $s$ where items $A$ and $B$ have already
> been taken and only items $C$ and $D$ remain. What is the $Q$-value of moving
> left, i.e. moving into square $B$?

*Exercise*:

> What is an example of a policy that achieves the highest return possible? Are
> there multiple such policies? If so which policies among these use the fewest
> possible moves? Which policies use the most possible moves?

For a variety of reasons, we often want to incentivize our agent to find
policies that use the fewest possible moves. We can often achieve this by adding
a constant penalty, i.e. negative reward, that the agent experiences on every move.

*Exercise*:

> Let's say that after every move the agent also accrues an additional -0.1
> reward no matter what the move was. What is an example of a policy that was
> optimal without this penalty but is no longer optimal with this penalty?

We also can use a discount factor $\gamma$ to incentivize policies that achieve
rewards more quickly. A discount factor means that when calculating the return
or Q-value, we multiply every future reward by $\gamma$. This is cumulative, so
that e.g. a reward achieved $n$ moves in the future is multipled by $\gamma ^n$.

This is called a discount factor because $\gamma$ is usually less than $1$,
which causes the agent to discount long-term future rewards in favor of
near-term rewards.

*Exercise*:

> Let's return to the policy that goes left, then up, then left, then up,
> etc. Let's assume we do *not* have the constant -0.1 per-move penalty. Instead
> we have a discount factor $\gamma$ of 0.9. What is the return of a game if we
> play according to this policy?

Having a discount factor and a constant per-move penalty can often achieve
similar aims, since practically speaking they often both incentivizing policies
that minimize the number of total moves, but they in fact do subtly different
things and so are usually used in conjunction with one another.

*Exercise*:

> What is an example of a policy in this game that is optimal with a -0.1
> per-move penalty but no discount factor which is no longer optimal if we
> remove the per-move penalty but reintroduce a discount factor, say of 0.9?

A poorly specified reward function is one major source of problems in
reinforcement learning, since it can lead to optimal policies that are not what
we actually wanted. As you likely found in the exercises, if we forget to
specify a per-move penalty, we can end up with an agent that learns to just
dilly-dally all day among empty squares.

More generally, as games and scenarios get more complex, we often find that
agents find policies which do technically maximize total reward, but do it in a
way that is completely alien and often undesirable. This is called "reward
hacking" and is one major worry we have about AI systems trained using
reinforcment learning.

For example, agents trained to play videogames where the score is the agent's
reward will often devise policies that take advantage of bugs in the videogame
to crank up the score rather than actually play the game.

This becomes an increasingly potent risks as agents have gotten better in the
last few years, since we've tasked them with increasingly complex tasks. These
tasks can become increasingly difficult to design watertight reward functions
for.

For example, if we task an agent with reducing global hunger by designing a
reward function that penalizes the agent for the length of time someone goes
hungry, we might hope that this incentivizes policies that feed people so that
they don't go hungry. However, an agent may instead conclude that the optimal
policy is simply to eliminate any human as soon that person goes hungry!

This becomes an even bigger problem when we realize that many reward functions
are very sparse. For example if we are using reinforcement learning to train an
agent to play chess, the most natural and simplest reward function we could is
to simply give a reward of $1$ if the agent checkmates its opponent and a reward
of $-1$ if the agent is checkmated with no reward for anything else.

This lets an agent "organically" learn the value of potentially sacrificing
pieces or taking pieces.

However, this is a very sparse set of rewards! That is almost always the agent
gets no reward and only at the very end of a game does the agent get reward one
way or the other. This can make trying to find the optimal policy extremely
difficult since we have to wait until the very end of a game to figure out
whether we've made a good move or a bad move.

Instead we often times must put in hacks/heuristics, euphemistically termed
"reward shaping" where we change our reward function away from its most natural
form to give more consistent and requent feedback to an agent. E.g. in chess we
might give large positive/negative rewards for checkmate/being checkmated, but
also give small positive/negative rewards for taking/losing pieces along the
way.

Of course reward shaping only heightens the chances of misspecifying our reward
function and ending up with reward hacking.

But we haven't yet actually discussed how to train an RL agent, which brings us
to...

# Pen-and-paper Q-learning

Now that we've explored some of the fundamental concepts of an RL setting, let's
actually train a model using RL. We'll use Q-learning here because it was one of
the early famous breakout hit of the deep learning revolution where deep neural
nets were used to train agents using Q-learning.

However, we will first train an agent using RL techniques via pen and paper to
get a better intuition for what role Bellman's equation plays in Q-learning.
Once we have this intuition, it is easier to see what role neural nets play.

To make this feasible, we will start with an extremely simple game, even simpler
than the previous game. We have a board with two cells laid side-by-side. They
are respectively cells `A` and `B`. The agent starts at a random cell at
the beginning of the game and can take turns deciding whether to go left or
right. No matter where you start, the objective of the game is to move to the
right side of the board, i.e. to get to the right-hand side of cell `B`.

This results in the following outcomes:

+ If the agent is at cell `A`:
    * Going left does nothing, the agent remains on cell `A` at the beginning of
      the next turn and receives a reward of -0.1
    * Going right moves the agent to cell `B` and the agent receives a reward of
      -0.1
+ If the agent is at cell `B`:
    * Going left moves the agent to cell `A` and the agent receives a reward of
      -0.1
    * Going right ends the game and the agent receives a reward of 1

Note that most moves (except the final move from cell `B` that ends the game)
have a mild negative reward penalty to encourage the agent to find the solution
with the fewest moves.

An ASCII representation of the board is given below.

```
-----
|A|B|
-----
```

We have two additional rules. There is a discount factor ($\gamma$) of 0.9.
Also, the game always ends after two turns no matter which cell the agent is
on.

*Exercise*

> Let's say our policy $\pi$ is to always go left, no matter what square we are
> at. If our current cell is cell $B$, what is the immediate reward I get after
> following this policy (i.e. what is the reward I get for performing one move)?
> What is $Q_\pi(B, right)$ where $Q_\pi(s, a)$ is the $Q$-value, i.e. total
> return, if every move performed after action $a$ is done according to the
> policy $\pi$? What is $Q_\pi(B, left)$? What is $Q_\pi(A, right)$?

*Exercise*

> Based on this description and reward function, what is the optimal policy for
> an agent to follow? I.e. what strategy should the agent use to maximize the
> total reward it receives for any given game?

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
A, right, turn 0 -> -1
A, left, turn 0 -> -3
B, right, turn 0 -> 4
B, left, turn 0 -> 0
A, right, turn 1 -> 0.2
A, left, turn 1 -> -0.1
B, right, turn 1 -> 1
B, left, turn 1 -> 2
```

*Exercise*:

> This game is simple enough that it is feasible to solve what the optimal $Q$
> function is directly without resorting to Q-learning via Bellman's equation to
> find what it is. What is the optimal $Q$ function? That is what is the $Q$
> value of each of the six possible state-action pairs if the agent is looking
> to maximize total reward at the end of a game?
>
> We can then compare what we get after running Q-learning to what we know the
> optimal $Q$ function should be.

Now use this optimal $Q$ function to determine what the agent's policy should
be.

Since this is the optimal $Q$ function, the optimal policy of the agent should
just be to do whatever action has the highest $Q$ value. You should find that
this means no matter which cell the agent is on, the agent should move right.
This should agree with your previous exercise result where you independently
came up with the optimal policy.

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

> Actually carry out all the above steps on your randomly initialized
> $Q$ function. Keep doing this until the $Q$ function stops changing no matter
> what state-action pair you use. You can do this in what order you want (just
> make sure that you end up iterating through every possible state-action pair,
> potentially multiple times). The exact number of times you will need to do
> this depends on the order you choose, but usually this will take about 6-12
> iterations.

You should find that no matter what starting values you chose for your $Q$
function, you end up with the values of the optimal $Q$ function you found
earlier.

As a reminder, what we've done is used Bellman's equation essentially as a
consistency check, and have learned the optimal $Q$ function by simply checking
whether our $Q$ function is consistent according to Bellman's equation (i.e.
whether the two sides of Bellman's equation equal each other) and updating our
$Q$ to force it to be consistent if not.

Deep Q-Learning (i.e. DQN for Deep Q-Network) follows the exact same general steps, but with neural
nets instead of a lookup table. So we'll start with a neural net with randomized
parameters that takes in a state-action pair and spits out a $Q$ value. We'll
calculate both sides of Bellman's equation as normal, but this time, to force
the two sides of Bellman's equation to be consistent, we'll take their
difference as a loss, and run gradient descent on the neural net's parameters to
reduce the loss. Then we repeat this process for new state-action pairs up to
some stopping point

Now let's move to some actual Python code to explore this in more detail!
