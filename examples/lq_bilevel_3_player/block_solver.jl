"""
This solver uses block arrays to compute the Stackelberg hierarchy equilibrium for a 3-player LQ game.
"""

# Import the game costs and dynamics from the problem definition.
include("problem.jl")

Zeros2 = zeros(2, 2)

"""
Solve the Stackelberg hierarchy for the followers (P2 and P3) at the terminal stage.
"""
# Construct M23 as a 10x10 block matrix.
M23 = BlockArray([
    [R[:,:,2] Zeros2 Zeros2 -B[:,player_control_list[2]]' Zeros2];
    [Zeros2 R[:,:,3] Zeros2 Zeros2 -B[:,player_control_list[3]]'];
    [Zeros2 Zeros2 Q[:,:,2] I(2) Zeros2];
    [Zeros2 Zeros2 Q[:,:,3] Zeros2 I(2)];
    [-B[:,player_control_list[2]] -B[:,player_control_list[3]] I(2) Zeros2 Zeros2]
], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2])

# Construct N23 as a 10x4 block matrix.
N23 = BlockArray([
    zeros(8, 4);                     # First 8 rows are zero
    [-A -B[:,player_control_list[1]]] # Last 2 rows: dynamics depend on x and u1
],
[8, 2],  # Row blocks
[2, 2]   # Column blocks (x and u1)
)
sol23 = -M23 \ N23 # P23 is the NE for players 2 and 3 for the terminal stage
println("second with blocks: ", sol23)

K2 = sol23[Block(1,1)]; # K2 is the feedback gain for player 2
P2 = sol23[Block(1,2)]; # P2 is the feedforward gain for player 2
K3 = sol23[Block(2,1)]; # K3 is the feedback gain for player 3
P3 = sol23[Block(2,2)]; # P3 is the feedforward gain for player 3
# NE: u2 = -K2 * x - P2 * u1
# NE: u3 = -K3 * x - P3 * u1


"""
Solve the Stackelberg hierarchy for the leader (P1) at the terminal stage.
"""
# Construct M1 as a 14x14 block matrix.
M1 = BlockArray([
    [R[:,:,1] Zeros2 Zeros2 Zeros2 -B[:,player_control_list[1]]'  -P2'   -P3' ];
    [Zeros2   Zeros2 Zeros2 Zeros2 -B[:,player_control_list[2]]'  I(2)  Zeros2];
    [Zeros2   Zeros2 Zeros2 Zeros2 -B[:,player_control_list[3]]' Zeros2  I(2) ];
    [Zeros2   Zeros2 Zeros2 Q[:,:,1]          I(2)               Zeros2 Zeros2];
    [-B[:,player_control_list[1]] -B[:,player_control_list[2]] -B[:,player_control_list[3]] I(2) Zeros2 Zeros2 Zeros2];
    [-P2      I(2)   Zeros2 Zeros2 Zeros2                        Zeros2 Zeros2];
    [-P3      Zeros2 I(2)   Zeros2 Zeros2                        Zeros2 Zeros2]
], [2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2])

# Construct N1 as a 14x2 block matrix.
N1 = BlockArray([
    zeros(8, 2); # First 8 rows are zero
    -A;
    -K2;
    -K3;
], [8, 2, 2, 2], [2]) # Row blocks and column blocks

sol1 = -M1 \ N1 # P1 is the NE for player 1 for the terminal stage
K1 = sol1[Block(1, 1)]
# NE for player 1: u1 = -K1 * x
println("first with blocks: ", K1)
