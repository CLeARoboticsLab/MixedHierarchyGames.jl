"""
Sets up a Stackelberg hierarchy problem for a three-player game with one leader of two followers.
Each follower has a Nash relationship with each other.

Drawing (at time t):
    P1
   /  \
  P2  P3

Each player has a pure quadratic cost function and planar single-integrator dynamics, 
i.e. state = 2D position and controls for each player are 2D velocity vectors.

# TODO: 
# 0. double check our transcribed math is correct;
# 1. add affine terms; 
# 2. try some different LQ game examples, compare with pure 3-player Nash;
# 3. generalize to 2-stage game.
"""

using LinearAlgebra
using BlockArrays


N = 3;

Q = zeros(2, 2, N);
Q[:, :, 1] = [1.0 0.1; 0.1 1.0];
Q[:, :, 2] = [1.0 0.2; 0.2 1.0];
Q[:, :, 3] = [1.0 0.3; 0.3 1.0];

R = zeros(2, 2, N);
R[:, :, 1] = [1.0 0.1; 0.1 1.0];
R[:, :, 2] = [1.0 0.2; 0.2 1.0];
R[:, :, 3] = [1.0 0.3; 0.3 1.0];

A = zeros(2, 2);
A = [1.0 0.0; 0.0 1.0];
B = zeros(2, 6);
B = [1.0 0.1 1.0 0.2 1.0 0.3; 
     0.0 1.0 0.0 1.0 0.0 1.0];


player_control_list = [
    [1,2],
    [3,4],
    [5,6]
];
