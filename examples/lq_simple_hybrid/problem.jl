"""
Sets up a Stackelberg hierarchy problem for a three-player game.
P2 leads P3, and both are in a Nash relationship with P1.

Drawing (at time t):
  P1  P2
      |
      P3

TODO: Loosen these assumptions as we generalize the solver.
Assumptions:
- Player controls are the same size.
- No constant terms in the dynamics or linear/constant terms in the cost functions.
- No state-control coupling costs (x Q u).
- Time invariant dynamics and costs.
- Player control costs are dependent only on the player's own control input.

# TODO: 
# 0. double check our transcribed math is correct;
# 1. add affine terms; 
# 2. try some different LQ game examples, compare with pure 3-player Nash;
# 3. generalize to 2-stage game.
"""

using LinearAlgebra
using BlockArrays

# Number of players
N = 3;

# State and control size.
mⁱ = 2; # control size per player

n = 2;      # total state size (2D position)
m = N * mⁱ; # total control size

# Define the cost matrices and dynamics for each player.
Q = zeros(n, n, N);
Q[:, :, 1] = [1.0 0.1; 0.1 1.0];
Q[:, :, 2] = [1.0 0.2; 0.2 1.0];
Q[:, :, 3] = [1.0 0.3; 0.3 1.0];

R = zeros(mⁱ, mⁱ, N);
R[:, :, 1] = [1.0 0.1; 0.1 1.0];
R[:, :, 2] = [1.0 0.2; 0.2 1.0];
R[:, :, 3] = [1.0 0.3; 0.3 1.0];

# Dynamics matrices for all players (n x n).
A = zeros(n, n);
A = [1.0 0.0; 0.0 1.0];

# Control matrices for all controls (n x m).
B = zeros(n, m);
B = [1.0 0.1 1.0 0.2 1.0 0.3;
     0.0 1.0 0.0 1.0 0.0 1.0];

player_control_list = [
    [1,2],
    [3,4],
    [5,6]
];

B1 = B[:, player_control_list[1]]
B2 = B[:, player_control_list[2]]
B3 = B[:, player_control_list[3]]
