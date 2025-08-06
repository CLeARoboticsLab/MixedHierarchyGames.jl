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
using Graphs

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


# Define the hierarchy tree.
roots = [1, 2];

#   (i,j) =  1 <-> i is parent of j.
#   (i,j) = -1 <-> i is child of j.
#   (i,i) =  ∞ <-> i is self.
#   (i,j) =  0 <-> i and j are Nash related.
G = zeros(N, N);
G[1, 1] = Inf;  # P1 is self
G[2, 2] = Inf;  # P2 is self
G[3, 3] = Inf;  # P3 is self

G[2, 3] = 1;  # P2 is parent of P3
G[3, 2] = -1; # P3 is child of P2

g = SimpleDiGraph(N);
add_edge!(g, 2, 3); # P2 -> P3

# TODO: sizing lookup currently assumes all players have the same control size.
var_sizes = Dict(
    'x' => n,  # state size
    'u' => mⁱ, # control size per player
    'λ' => n,  # dynamics dual size
    # 'ψ' => m,  # control dual size
    'η' => mⁱ,  # leader-follower policy dual size
)

function lookup_varsize(var_name::Symbol)
    symbol_str = string(var_name)
    return var_sizes[symbol_str[1]]
end
