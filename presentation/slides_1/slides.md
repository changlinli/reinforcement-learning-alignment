% Reinforcement Learning
% Changlin Li (mail@changlinli.com)

# Following up from last time

+ Created an agent with really bad inner alignment issues/goal misgeneralization
    * In training avoids humans
    * In certain production environments makes a beeline for humans!

# How can we try to solve this problem?

Solving goal misgeneralization is one of the *core* questions of AI alignment

+ No singularly convincing solution yet
+ A lot of different bets that might work or might not
+ Today we'll look at one

# Mechanistic Interpretation

+ The idea: find a coherent way of understanding what individual neurons or
  sections of a neural net are doing
    * This hopefully lets you "read the neural net's mind" to see if it might be
      misbehaving
+ The bet: different neural nets tend to converge on a similar set of structures
    * If we build a library of these structures and develop a good understanding
      of them, we can build widely generalizable understanding of neural nets

# What we'll be coding today

+ Not the entirety of mechanistic interpretation (too big!)
+ Looking at one tool that shows up a lot in the mechanistic interpretation
  toolbox: "Dreaming"
    * If people want, can use this tool to then examine individual neurons, but
      we won't do that here

# Dreaming: running a neural net in reverse

+ Can we get the neural net to tell us what mazes will cause it to decide to
  harvest humans?

# Thinking back to neural net training

Neural net training and inference is all about choosing to fix certain elements
as constant and perhaps let other elements vary according to some mathematical
formula.

+ Inference: hold the input constant and the neural net's weights+biases
  constant and calculate the output
+ Training: hold the input constant, let the neural net's weights+biases vary
  based on gradient descent on some loss function

# Gradient descent works without caring about what's constant and what varies

What if we just change what we hold constant and vary?

+ Hold the *weights+biases* constant
+ Vary the *input* based on gradient descent on some loss function
+ Let the loss function be how close we are to some pathological behavior

# Still an optimization problem!

All the same tools still work! We are asking the same question with different
parameters:

+ Usual training question: what set of weights+biases gets me closest to the
  correct answer?
+ Our new question: what set of inputs gets me closest to the wrong answer?

# Deep dreaming

This principle of getting neural nets to generate inputs that fulfill some
loss function is called "deep dreaming" (deep for deep neural nets).

# Applying the principle to the problem at hand

+ Our pathological behavior/wrong answer: standing in the square next to a human
  and assigning a high Q-value to moving onto the square with the human.
+ Our input: the maze
+ Because we are trying to maximize our Q-value we will be using gradient ascent
  rather than descent (you could make it descent simply by flipping signs, if
  you wanted)
+ **Using this can we get advance warning of what mazes will cause our bot to
  harvest humans before unleashing it into the wild?**

# Current representation of the maze

```
example_maze = torch.tensor([
    [1, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 2, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0],
    [3, 1, 1, 1, 1, 1, -1],
])
```

# Problems with our current representation

+ Our current representation of the maze is ill-suited for the problem
+ In general we want whatever we vary to be *smooth*
    * Current representation is very discrete and hence not smooth
    * What `0` means is completely different from what `1` means, completely
      different from what `2` means, etc.
    * Intermediate values don't make sense (if while varying we end up with a
      `0.5` what does that mean? Kind of a wall? Sort of a crop? Just a little
      bit of a person?)
    * Analogous to problems with using a single output neuron for digit
      classification instead of ten outputs, one for `0`-`9`

# Tweaking the representation

+ Let's take the same solution for digit classification:
    * Split out what was one value into multiple values
    * Same intuition behind one-hot encodings
+ Previously: One 7x7 maze with different values for walls, spaces, crops,
  humans, and a finishing square
+ Now: Four 7x7 "blocks": one for wall locations, one for crop locations, one for
  human locations, and one for the finish location
+ See `main_train.to_input` for the conversion

# Same maze from previously: wall locations

```
wall_locations = torch.tensor([
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0],
])
```

# Same maze from previously: human locations

```
human_locations = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
])
```

# Same maze from previously: crop locations

```
crop_locations = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
])
```

# Same maze from previously: finish locations

```
finish_locations = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
])
```

# New representation is smoother

+ Intermediate values make sense now
    * What does `0.5` in the wall location block mean?
        - A wall there weakly contributes to pathological behavior

# Our new representation still demonstrates goal misgeneralization!

+ Look at the examples at the end of `main_train.py`
    * Bot still makes a beeline for humans in certain mazes!
+ So our new representation is every bit as capable (and misguided) as our old
  representation

# The task at hand

+ Keep everything except the wall locations constant
+ Allow the wall locations to vary
+ Calculate the gradient for how much the wall locations should vary to maximize
  the Q-value of an agent 

# Let's write the code

+ In `maze_dream.py`, there are some TODOs in `wall_locations_gradient`
    * Fill them in and the rest of the code should just run out of the box
+ This won't be a lot of code (probably less than 10 lines total to add)
    * If we finish early can go on to individual neurons!
