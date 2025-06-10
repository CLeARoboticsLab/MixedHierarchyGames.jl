"""
Sets up a Stackelberg hierarchy problem for a three-player game with one leader of two followers.
Each follower has a Nash relationship with each other.

Drawing (at time t):
    P1
   /  \
  P2  P3

Each player has a pure quadratic cost function and planar single-integrator dynamics, 
i.e. state = 2D position and controls for each player are 2D velocity vectors.

Assumptions:
- Player controls are the same size.
- No constant terms in the dynamics or linear terms in the cost functions.

# TODO: 
# 0. double check our transcribed math is correct;
# 1. add affine terms; 
# 2. try some different LQ game examples, compare with pure 3-player Nash;
# 3. generalize to 2-stage game.
"""

using LinearAlgebra
using BlockArrays
using BlockDiagonals

N = 3;

# State and control size.
nⁱ = 2; # per-player state size
mⁱ = 2; # control size per player

n = N * nⁱ; # total state size
m = N * mⁱ; # total control size

# Note: this is a decoupled optimization problem where each player costs their own portion of the state.
# TODO: Adjust the costs to introduce coupling between players.
Q = zeros(n, n, N);
Q[1:nⁱ, 1:nⁱ, 1] = [1.0 0.1; 0.1 1.0];
Q[nⁱ+1:2*nⁱ, nⁱ+1:2*nⁱ, 2] = [1.0 0.2; 0.2 1.0];
Q[2*nⁱ+1:3*nⁱ, 2*nⁱ+1:3*nⁱ, 3] = [1.0 0.3; 0.3 1.0];

R = zeros(mⁱ, mⁱ, N);
R[:, :, 1] = [1.0 0.1; 0.1 1.0];
R[:, :, 2] = [1.0 0.2; 0.2 1.0];
R[:, :, 3] = [1.0 0.3; 0.3 1.0];


A = zeros(n, n);
Aⁱ = [1.0 0.0; 0.0 1.0];
A = BlockDiagonal([Aⁱ, Aⁱ, Aⁱ]); # 3 players, each with 2D state

B = zeros(n, mⁱ, N);
B¹ = [1.0 0.1; 0.0 1.0]; # Player 1 control matrix is costed based on its own control.
B² = [1.0 0.2; 0.0 1.0]; # Player 2 control matrix is costed based on its own control.
B³ = [1.0 0.3; 0.0 1.0]; # Player 3 control matrix is costed based on its own control.
B[:, :, 1] = BlockArray([B¹; zeros(nⁱ, mⁱ); zeros(nⁱ, mⁱ)], [nⁱ, nⁱ, nⁱ], [mⁱ]);
B[:, :, 2] = BlockArray([zeros(nⁱ, mⁱ); B²; zeros(nⁱ, mⁱ)], [nⁱ, nⁱ, nⁱ], [mⁱ]);
B[:, :, 3] = BlockArray([zeros(nⁱ, mⁱ); zeros(nⁱ, mⁱ); B³], [nⁱ, nⁱ, nⁱ], [mⁱ]);


player_control_list = [
    [1,2],
    [3,4],
    [5,6]
];
