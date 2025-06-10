"""
This solver manually computes the Stackelberg hierarchy equilibrium for a 3-player LQ game, by solving for each player
i's controls at each time t and using the solution to build the KKT conditions at the next time.
"""

# Import the game costs and dynamics from the problem definition.
include("problem.jl")

"""
Solve the Stackelberg hierarchy for the followers (P2 and P3) at the terminal stage.
"""
# BUG: state/Q^i_t should be 6x6, not 2x2 because it is multiplied by the whole state.
M23 = zeros(10, 10); # (2 + 2) + (2) + (2) + (2) = 10
# rows 1:4
M23[1:2, 1:2] = R[:,:,2];
M23[3:4, 3:4] = R[:,:,3];
M23[1:2, 7:8] = -B[:,player_control_list[2]]';
M23[3:4, 9:10] = -B[:,player_control_list[3]]';
# rows 5:8
M23[5:6, 5:6] = Q[:,:,2];
M23[7:8, 5:6] = Q[:,:,3];
M23[5:8, 7:10] = I(4);
# rows 9:10
M23[9:10, 1:2] = -B[:,player_control_list[2]]; # u2
M23[9:10, 3:4] = -B[:,player_control_list[3]]; # u3
M23[9:10, 5:6] = I(2);

N23 = zeros(10, 4);
N23[9:10, 1:2] = -A;
N23[9:10, 3:4] = -B[:,player_control_list[1]]; # u2

sol23 = -M23 \ N23 # P23 is the NE for players 2 and 3 for the terminal stage
println("first: ", sol23)
K2 = sol23[1:2, 1:2]; # K2 is the feedback gain for player 2
P2 = sol23[1:2, 3:4]; # P2 is the feedforward gain for player 2
K3 = sol23[3:4, 1:2]; # K3 is the feedback gain for player 3
P3 = sol23[3:4, 3:4]; # P3 is the feedforward gain for player 3
# NE: u2 = -K2 * x - P2 * u1
# NE: u3 = -K3 * x - P3 * u1

"""
Solve the Stackelberg hierarchy for the leader (P1) at the terminal stage.
"""
# terminal stage, player 1:
M1 = zeros(14, 14);
# rows 1:6
M1[1:2, 1:2] = R[:,:,1];
M1[1:2, 9:10] = -B[:,player_control_list[1]]';
M1[1:2, 11:12] = -P2';
M1[1:2, 13:14] = -P3';
M1[3:4, 9:10] = -B[:,player_control_list[2]]';
M1[3:6, 11:14] = I(4);
M1[5:6, 9:10] = -B[:,player_control_list[3]]';
# rows 7:8
M1[7:8, 7:8] = Q[:,:,1];
M1[7:8, 9:10] = I(2);
# rows 9:10
M1[9:10, 1:2] = -B[:,player_control_list[1]]; # u1
M1[9:10, 3:4] = -B[:,player_control_list[2]]; # u2
M1[9:10, 5:6] = -B[:,player_control_list[3]]; # u3
M1[9:10, 7:8] = I(2);
# rows 11:12
M1[11:12, 1:2] = -P2;
M1[11:12, 3:4] = I(2);
# rows 13:14
M1[13:14, 1:2] = -P3;
M1[13:14, 5:6] = I(2);

# N1 matrix for player 1:
N1 = zeros(14, 2);
N1[9:10, :] = -A;
N1[11:12, :] = -K2;
N1[13:14, :] = -K3;

sol1 = -M1 \ N1 # P1 is the NE for player 1 for the terminal stage
K1 = sol1[1:2, 1:2]; # K1 is the feedback gain for player 1
# NE for player 1: u1 = -K1 * x
println("first with manual: ", K1)