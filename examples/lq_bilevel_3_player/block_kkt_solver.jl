"""
This solver manually computes the Stackelberg hierarchy equilibrium for a 3-player LQ game, by appending each players'
KKT conditions at each time to the higher-level players' KKT conditions and at earlier times.

Note: We use ordering mathcal{Z}ⁱₜ based on page 4 of Jingqi's goodnotes, which is different from the ordering used in block_solver.jl.
mathcal{Z}ⁱₜ = [player i primal at t;
                player i dual (dynamics + policy) at t;
                future duals (Nash at t);
                future primals + duals (followers) at t;
                future player KKT states at t, zⁱₜ]
Question: What do we do when we have multiple players in a more complex hierarchy?

Note: For the information vector, we use the ordering
mathcal{Y}ⁱₜ = [xₜ, u¹ₜ] for followers P2 and P3, and mathcal{Y}¹ₜ = [xₜ] for leader P1.
"""

# Import the game costs and dynamics from the problem definition.
include("problem.jl")

"""
Solve the Stackelberg hierarchy for the followers (P2 and P3) at the terminal stage. This should not be different from
the KKT conditions in block_solver.jl, though ordering and sizing may differ.
"""
# For stage t=T, we require primal variables for the controls of P2 and P3 (u²ₜ, u³ₜ) and the final state xₜ₊₁.
# We also require dual variables λ²ₜ and λ³ₜ for the dynamics constraints of P2 and P3.

# 1. We first define the ordering \mathcal{Z}ⁱₜ of the state zⁱₜ at time t=T for players 2 and 3.
# \mathcal{Z}²³ₜ = [ u²ₜ, u³ₜ, λ²ₜ, λ³ₜ, xₜ₊₁ ].

# 2. Compute the size of the stage state zⁱₜ for players 2 and 3 at time t=T.
# (2 + 2) + (2) + (2) + (2) = 10 <=> 2 player controls, 2 dynamics constraints, 1 full state
z_sizes²³ₜ = [mⁱ, mⁱ, n, n, n];
s²³ₜ = sum(z_sizes²³ₜ);

# 3. Identify the ordering \mathcal{Y}ⁱₜ of the information vector (i.e. state and leader information that P2 and P3
#    use for decision-making) at t=T.
# \mathcal{Y}²³ₜ = [ xₜ, u¹ₜ ].

# 4. Compute the size s̃ⁱₜ of the information vector yⁱₜ for players 2 and 3 at time t=T.
# (2) + (2) = 4 <=> 1 full state, 1 player control
w_sizes²³ₜ = [n, mⁱ];
s̃²³ₜ = sum(w_sizes²³ₜ);

# 5. Compute the M matrix for P2 and P3 at final stage t=T.
M23 = zeros(s²³ₜ, s²³ₜ);

Zeros2 = zeros(2, 2)
I2 = I(n)
# Construct M23 as a 10x10 block matrix.
M23 = BlockArray([
    [R[:,:,2] Zeros2 -B[:,player_control_list[2]]' Zeros2 Zeros2];
    [Zeros2 R[:,:,3] Zeros2 -B[:,player_control_list[3]]' Zeros2];
    [Zeros2 Zeros2 I2 Zeros2 Q[:,:,2]];
    [Zeros2 Zeros2 Zeros2 I2 Q[:,:,3]];
    [-B[:,player_control_list[2]] -B[:,player_control_list[3]] Zeros2 Zeros2 I2]
], z_sizes²³ₜ, z_sizes²³ₜ)

# 6. Compute the N matrix for P2 and P3 at final stage t=T.
N23 = BlockArray([
    zeros(s²³ₜ - n, s̃²³ₜ); # First 8 rows are zero
    -A -B[:,player_control_list[1]] # u1
], z_sizes²³ₜ, w_sizes²³ₜ)

# 7. [skipped due to assumptions] Compute n23 block vector for P2 and P3 at final stage t=T.

# 8. Compute the solution matrices for P2 and P3 at final stage t=T, i.e. for comparison.
sol23 = -M23 \ N23; # P23 is the NE for players 2 and 3 for the terminal stage
# println("second with blocks: ", sol23)

K2 = sol23[Block(1,1)]; # K2 is the feedback gain for player 2
P2 = sol23[Block(1,2)]; # P2 is the feedforward gain for player 2
K3 = sol23[Block(2,1)]; # K3 is the feedback gain for player 3
P3 = sol23[Block(2,2)]; # P3 is the feedforward gain for player 3
# NE: u2 = -K2 * x - P2 * u1
# NE: u3 = -K3 * x - P3 * u1

"""
Solve the Stackelberg hierarchy for the leader (P1) at the terminal stage.
"""
# For stage t=T, we require primal variables for the controls of all players (u¹ₜ, u²ₜ, u³ₜ) and the final state xₜ₊₁.
# We also require dual variables λ¹ₜ, λ²ₜ, and λ³ₜ for the dynamics constraints of each player, and 
# ψ¹⁻²ₜ, ψ¹⁻³ₜ for the policy constraints which tie P1's problem at time t=T to that of followers P2 and P3.

# 1. We first define the ordering \mathcal{Z}¹ₜ of the state z¹ₜ at time t=T for player 1.
# \mathcal{Z}¹ₜ = [ u¹ₜ, u²ₜ, u³ₜ, xₜ₊₁, λ²ₜ, λ³ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ ] = [ u¹ₜ, z²³ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ ].
# \mathcal{Z}¹ₜ = [ u¹ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ, u²ₜ, u³ₜ, λ²ₜ, λ³ₜ, xₜ₊₁ ] = [ u¹ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ, z²³ₜ ].
# We note that the ordering of z¹ₜ is a concatenation of the control of P1, P1's dual variables λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ, and
# the full state z²³ₜ of P2 and P3.

# 2. Compute the size of the stage state z¹ₜ for player 1 at time t=T.
# (2) + (2 + 2 + 2) + (10) = 18 <=> P1 controls, 1 dynamics, 2 policy, full KKT vector of P2 and P3 at time t=T.
z_sizes¹ₜ = [mⁱ, n, mⁱ, mⁱ, s²³ₜ];
z_sizes¹ₜ = vcat([mⁱ, n, mⁱ, mⁱ], z_sizes²³ₜ);
s¹ₜ = sum(z_sizes¹ₜ);

# 3. Identify the ordering \mathcal{Y}¹ₜ of the information vector (i.e. state and leader information that P1
#    use for decision-making) at t=T.
# \mathcal{Y}¹ₜ = [ xₜ ].

# 4. Compute the size s̃¹ₜ of the information vector yⁱₜ for player 1 at time t=T.
# (2) = 2 <=> 1 state
w_sizes¹ₜ = [n];
s̃¹ₜ = sum(w_sizes¹ₜ);

# 5. Compute the M1 matrix (18x18) for P1 at final stage t=T.
M1 = BlockArray(zeros(s¹ₜ, s¹ₜ), z_sizes¹ₜ, z_sizes¹ₜ);

# First block row of KKT conditions corresponds to gradient of Lagrangian with respect to u¹ₜ.
M1[Block(1,1)] = R[:,:,1]; # Player 1's cost
M1[Block(1,2)] = -B[:,player_control_list[1]]'; # Player 1's control input
M1[Block(1,3)] = -P2'; # Player 2's feedforward gain
M1[Block(1,4)] = -P3'; # Player 3's feedforward gain

# Second block row of KKT conditions corresponds to gradient of Lagrangian with respect to u²ₜ.
M1[Block(2,2)] = -B[:,player_control_list[2]]'; # Player 2's control input
M1[Block(2,3)] = I(mⁱ);                         # Player 2's policy gradient ψ¹²ₜ

# Third block row of KKT conditions corresponds to gradient of Lagrangian with respect to u³ₜ.
M1[Block(3,2)] = -B[:,player_control_list[3]]'; # Player 3's control input
M1[Block(3,4)] = I(mⁱ);                         # Player 3's policy gradient ψ¹³ₜ

# Fourth row of KKT conditions corresponds to gradient of Lagrangian with respect to xₜ₊₁.
M1[Block(4,2)] = I(n);                          # Player 1's dynamics λ¹ₜ
last_block_col = length(z_sizes¹ₜ);              # Last column index for z¹ₜ
M1[Block(4, last_block_col)] = Q[:,:,1];        # Player 1's state cost

# Main.@infiltrate
# Fifth block row of KKT conditions corresponds to gradient of Lagrangian with respect to z²³ₜ.
M1[Block.(5:last_block_col), Block(1)] = N23[Block.(1:5), Block(2)];    # Player 2's control input
M1[Block.(5:last_block_col),Block.(5:last_block_col)] = M23                            # KKT conditions for P2 and P3

# 6. Compute the N1 matrix (18x2) for P1 at final stage t=T.
N1 = BlockArray([
    zeros(s¹ₜ - n, s̃¹ₜ); # First 16 rows are zero
    -A
], z_sizes¹ₜ, w_sizes¹ₜ)

# 7. [skipped due to assumptions] Compute n23 block vector for P2 and P3 at final stage t=T.

# 8. Compute the solution matrices for P1 at final stage t=T, i.e. for comparison.
sol1 = -M1 \ N1; # P1 is the NE for players 1, 2, and 3 for the terminal stage
# println("second with blocks: ", sol23)

K1 = sol1[Block(1,1)]; # K1 is the feedback gain for player 1.
# NE: u1 = -K1 * x


# TODO: Write code for time T-1 for all players.
