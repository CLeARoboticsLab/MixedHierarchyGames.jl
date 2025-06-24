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
M23 = BlockArray(
    zeros(10, 10)
, [2, 2, 2, 2, 2], [2, 2, 2, 2, 2])
M23[Block(1,1)] = R[:,:,2]; # Player 2's cost
M23[Block(1,3)] = -B[:,player_control_list[2]]'; # Player 2's control input

M23[Block(2,2)] = R[:,:,3]; # Player 3's cost
M23[Block(2,4)] = -B[:,player_control_list[3]]'; # Player 3's control input

M23[Block(3,3)] = I(2);
M23[Block(3,5)] = Q[:,:,2]; # Player 2's state cost

M23[Block(4,4)] = I(2);
M23[Block(4,5)] = Q[:,:,3]; # Player 3's state cost

M23[Block(5,1)] = -B[:,player_control_list[2]]; # Player 2's control input
M23[Block(5,2)] = -B[:,player_control_list[3]]; # Player 3's control input
M23[Block(5,5)] = I(2); # Player 2's state cost




# Construct N23 as a 10x4 block matrix.
N23 = BlockArray(zeros(10, 4),
[8, 2],  # Row blocks
[2, 2]   # Column blocks (x and u1)
)
N23[Block(2,1)] = -A; # Dynamics for player 2
N23[Block(2,2)] = -B[:, player_control_list[1]]; # Dynamics for player 3

sol23 = -M23 \ N23 # P23 is the NE for players 2 and 3 for the terminal stage
# println("second with blocks: ", sol23)

K2 = sol23[Block(1,1)]; # K2 is the state gain for player 2
P2 = sol23[Block(1,2)]; # P2 is the u1-gain 
K3 = sol23[Block(2,1)]; # K3 is the state gain for player 3
P3 = sol23[Block(2,2)]; # P3 is the u2-gain for player 3
# NE: u2 = -K2 * x - P2 * u1
# NE: u3 = -K3 * x - P3 * u1


"""
Solve the Stackelberg hierarchy for the leader (P1) at the terminal stage.
"""
# state dimension: 

# Construct M1 as a 14x14 block matrix.
M1 = BlockArray(zeros(18, 18), [2, 2, 2, 2, 10], [2, 2, 2, 2, 2, 2, 2, 2, 2]) # Row blocks and column blocks

M1[Block(1,1)] = R[:,:,1]; # Player 1's cost
M1[Block(1,2)] = -B[:,player_control_list[1]]'; # Player 1's control input
M1[Block(1,3)] = -P2'; # Player 2's feedforward gain
M1[Block(1,4)] = -P3'; # Player 3's feedforward gain
M1[Block(2,2)] = -B[:,player_control_list[2]]'; # Player 2's control input
M1[Block(2,3)] = I(2);
M1[Block(3,2)] = -B[:,player_control_list[3]]'; # Player 3's control input
M1[Block(3,4)] = I(2);
M1[Block(4,2)] = I(2);
M1[Block(4,9)] = Q[:,:,1]; # Player 1's state cost

M1[Block(5,1)] = N23[:, [3,4]];
M1[9:end, 9:end] = M23;


# Construct N1 as a 14x2 block matrix.
N1 = BlockArray(zeros(18,2), [16, 2], [2]) # Row blocks and column blocks
N1[Block(2,1)] = -A;

sol1 = -M1 \ N1 # P1 is the NE for player 1 for the terminal stage
K1 = sol1[Block(1, 1)]
# NE for player 1: u1 = -K1 * x
# println("first with blocks: ", K1)





B1 = B[:, player_control_list[1]]; # Player 1's control input
B2 = B[:, player_control_list[2]]; # Player 2's control input
B3 = B[:, player_control_list[3]]; # Player 3's control input


# principled construction of players 2 and 3 for the stage T-1

# total state dimension: 18 + 18 = 36

D23_1 = BlockArray(zeros(18, 18), [2, 2, 4, 8, 2], [2, 2, 4, 8, 2])
D23_2 = BlockArray(zeros(18, 36 - 18), [2, 2, 4, 8, 2], [6*2, 4, 2])


D23_1[Block(1,1)] = R[:,:,2]; # Player 2's cost
D23_1[Block(1,3)] = [-B[:,player_control_list[2]]' zeros(2,2)]; # Player 2's control input
D23_1[Block(2,2)] = R[:,:,3]; # Player 3's cost
D23_1[Block(2,3)] = [zeros(2,2) -B[:,player_control_list[3]]']; # Player 3's control input

D23_1[Block(3,3)] = I(4);
D23_1[Block(3,4)] = -[K1' K3' zeros(2,4); zeros(2,4) K1' K2'];
D23_1[Block(3,5)] = [Q[:,:,2]; Q[:,:,3]];

D23_1[Block(4,4)] = [
    I(2) -P2' zeros(2,4);
    zeros(2,2) I(2) zeros(2,4);
    zeros(2,4) I(2) -P3';
    zeros(2,6) I(2)  
];

D23_1[Block(5,1)] = -B[:,player_control_list[2]]; # Player 2's control input
D23_1[Block(5,2)] = -B[:,player_control_list[3]]; # Player 3's control input
D23_1[Block(5,5)] = I(2); # Player 2's state cost




D23_2[Block(3,2)] = -[A' zeros(2,2); zeros(2,2) A'];
D23_2[Block(4,2)] = -[B1 B3 zeros(2,4); zeros(2,4) B1 B2]';




MM23 = BlockArray(zeros(18 + 18, 36), [18, 18], [18, 18])
MM23[Block(1,1)] = D23_1;
MM23[Block(1,2)] = D23_2; # Player 2's and 3's control input
MM23[Block(2,1)] = [zeros(18, 16) N1]
MM23[Block(2,2)] = M1; # Player 2's and 3's control input


NN23 = BlockArray(zeros(36, 4), [16, 2, 18], [2, 2])
NN23[Block(2,1)] = -A;
NN23[Block(2,2)] = -B1; # Dynamics for player 2


ssol23 = -MM23 \ NN23;

KK2 = ssol23[1:2, 1:2]; # K2 is the feedback gain for player 2
PP2 = ssol23[1:2, 3:4]; # P2 is the feedforward gain for player 2
KK3 = ssol23[1:2, 1:2]; # K3 is the feedback gain for player 3
PP3 = ssol23[1:2, 3:4]; # P3 is the feedforward gain for player 3



