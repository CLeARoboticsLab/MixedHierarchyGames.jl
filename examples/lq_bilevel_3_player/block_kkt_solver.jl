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

Dual variables:
λ - dynamics
ψ - intra-stage leadership
η - inter-stage policy (between leaves and next stage)

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
    # first row is the gradient of the Lagrangian with respect to u²ₜ
    [R[:,:,2] Zeros2 -B[:,player_control_list[2]]' Zeros2 Zeros2];
    # second row is the gradient of the Lagrangian with respect to u³ₜ
    [Zeros2 R[:,:,3] Zeros2 -B[:,player_control_list[3]]' Zeros2];
    # third row is the gradient of the Lagrangian of P2 with respect to xₜ₊₁
    [Zeros2 Zeros2 I2 Zeros2 Q[:,:,2]];
    # fourth row is the gradient of the Lagrangian of P3 with respect to xₜ₊₁
    [Zeros2 Zeros2 Zeros2 I2 Q[:,:,3]];
    # fifth row is the gradient of both Lagrangian with respect to λₜ (i.e. combined dynamics b/c they are same)
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
# TODO:  NE: u2 = -K2 * x - P2 * u1,  NE: u3 = -K3 * x - P3 * u1
# Current:  NE: u2 = K2 * x + P2 * u1,  NE: u3 = K3 * x + P3 * u1

"""
Solve the Stackelberg hierarchy for the leader (P1) at the terminal stage.
"""
# For stage t=T, we require primal variables for the controls of all players (u¹ₜ, u²ₜ, u³ₜ) and the final state xₜ₊₁.
# We also require dual variables λ¹ₜ, λ²ₜ, and λ³ₜ for the dynamics constraints of each player, and 
# ψ¹⁻²ₜ, ψ¹⁻³ₜ for the policy constraints which tie P1's problem at time t=T to that of followers P2 and P3.

# 1. We first define the ordering \mathcal{Z}¹ₜ of the state z¹ₜ at time t=T for player 1.
# [not this] \mathcal{Z}¹ₜ = [ u¹ₜ, u²ₜ, u³ₜ, xₜ₊₁, λ²ₜ, λ³ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ ] = [ u¹ₜ, z²³ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ ].
# [this is correct] \mathcal{Z}¹ₜ = [ u¹ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ, u²ₜ, u³ₜ, λ²ₜ, λ³ₜ, xₜ₊₁ ] = [ u¹ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ, z²³ₜ ].
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

# TODO: Change these to use the standard control law u = -Kx - Pu
M1[Block(1,3)] = -P2'; # Player 2's feedforward gain (negative assuming u2 = K2 * x + P2 * u1)
M1[Block(1,4)] = -P3'; # Player 3's feedforward gain (positive assuming u3 = K3 * x + P3 * u1)

# Second block row of KKT conditions corresponds to gradient of Lagrangian with respect to u²ₜ.
M1[Block(2,2)] = -B[:,player_control_list[2]]'; # Player 2's control input
M1[Block(2,3)] = I(mⁱ);                         # Player 2's policy gradient ψ¹²ₜ

# Third block row of KKT conditions corresponds to gradient of Lagrangian with respect to u³ₜ.
M1[Block(3,2)] = -B[:,player_control_list[3]]'; # Player 3's control input
M1[Block(3,4)] = I(mⁱ);                         # Player 3's policy gradient ψ¹³ₜ

# Fourth row of KKT conditions corresponds to gradient of Lagrangian of P1 with respect to xₜ₊₁.
M1[Block(4,2)] = I(n);                          # Player 1's dynamics λ¹ₜ
last_block_col = length(z_sizes¹ₜ);              # Last column index for z¹ₜ
M1[Block(4, last_block_col)] = Q[:,:,1];        # Player 1's state cost

# Note: dynamics constraint exists in the KKT conditions for P2 and P3, so we don't need to rewrite it here.

# Fifth block row of KKT conditions corresponds to gradient of Lagrangian with respect to z²³ₜ.
M1[Block.(5:last_block_col), Block(1)] = N23[Block.(1:5), Block(2)];    # Player 2's control input
M1[Block.(5:last_block_col), Block.(5:last_block_col)] = M23            # KKT conditions for P2 and P3

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
# TODO: NE: u1 = -K1 * x
# Current: NE: u1 = K1 * x


"""
Solve the Stackelberg hierarchy for P2 and P3 at the penultimate stage (t=T-1).
"""
# 1. We first define the ordering \mathcal{Z}ⁱₜ of the state zⁱₜ at time t=T-1 for players 2 and 3.
# \mathcal{Z}²³ₜ = [ u²ₜ, u³ₜ, λ²ₜ, λ³ₜ, η²⁻¹ₜ₊₁, η²⁻³ₜ₊₁, η³⁻¹ₜ₊₁, η³⁻²ₜ₊₁, xₜ₊₁, z¹ₜ₊₁ ].

# 2. Compute the size of the stage state zⁱₜ for players 2 and 3 at time t=T-1.
# (2+2) + (2+2) + (2+2+2+2) + (2) + (18) = 36
zz_sizes²³ₜ = vcat([mⁱ, mⁱ, n, n, mⁱ, mⁱ, mⁱ, mⁱ, n], z_sizes¹ₜ);
ss²³ₜ = sum(zz_sizes²³ₜ);

# TODO: This is consistent between stages, so can be done once at the beginning. Refactor to make it so.
# 3. Identify the ordering \mathcal{Y}ⁱₜ of the information vector (i.e. state and leader information that P2 and P3
#    use for decision-making) at t=T.
# \mathcal{Y}²³ₜ = [ xₜ, u¹ₜ ].

# TODO: This is consistent between stages, so can be done once at the beginning. No need to redefine variables.
# 4. Compute the size s̃ⁱₜ of the information vector yⁱₜ for players 2 and 3 at time t=T.
# (2) + (2) = 4 <=> 1 full state, 1 player control
ww_sizes²³ₜ = [n, mⁱ];
s̃s²³ₜ = sum(ww_sizes²³ₜ);

# 5. Compute the MM23 matrix (36x36) for P2 and P3 at penultimate stage t=T-1.
MM23 = BlockArray(zeros(ss²³ₜ, ss²³ₜ), zz_sizes²³ₜ, zz_sizes²³ₜ);

# 6. In addition, we set up the NN23 matrix (36x4) for P2 and P3 at penultimate stage t=T-1.
NN23 = BlockArray(zeros(ss²³ₜ, s̃s²³ₜ), zz_sizes²³ₜ, ww_sizes²³ₜ);

# Row 1 corresponds to the gradient of the P2 Lagrangian with respect to u²ₜ.
MM23[Block(1,1)] = R[:,:,2]; # u²ₜ
MM23[Block(1,3)] = -B[:,player_control_list[2]]'; # λ²ₜ

# Row 2 corresponds to the gradient of the P3 Lagrangian with respect to u³ₜ.
MM23[Block(2,2)] = R[:,:,3]; # u³ₜ
MM23[Block(2,4)] = -B[:,player_control_list[3]]'; # λ³ₜ

# Row 3 corresponds to the dynamics equality constraint (primal feasibility condition).
# We do not compute separate gradients for P2 wrt λ²ₜ and P3 wrt to λ³ₜ because dynamics are a shared constraint 
# over the joint state. This is true of any constraint which is coupled.

# Row 3 corresponds to the gradient of the P2 Lagrangian with respect to λ²ₜ (shared dynamics).
MM23[Block(3,1)] = -B[:,player_control_list[2]]; # u²ₜ
MM23[Block(3,2)] = -B[:,player_control_list[3]]; # u³ₜ
MM23[Block(3,9)] = I(n);                          # xₜ

NN23[Block(3,1)] = -A; # xₜ
NN23[Block(3,2)] = -B[:,player_control_list[1]]; # u¹ₜ

# # Row 4 corresponds to the gradient of the P3 Lagrangian with respect to λ³ₜ (duplicated dynamics),
# # which is the same in this problem as row 3.
# MM23[Block(4), Block.(1:length(zz_sizes²³ₜ))] = MM23[Block(3), Block.(1:length(zz_sizes²³ₜ))];
# NN23[Block(4), Block.(1:length(ww_sizes²³ₜ))] = NN23[Block(3), Block.(1:length(ww_sizes²³ₜ))];

# TODO: Create a lookup table for blocks based on variable and use it here so we can avoid these issues for earlier times.
# Row 4 corresponds to the gradient of the P2 Lagrangian with respect to u¹ₜ₊₁.
MM23[Block(4,5)] = I(mⁱ); # η²⁻¹ₜ₊₁

# TODO:  NE: u2 = -K2 * x - P2 * u1,  NE: u3 = -K3 * x - P3 * u1
# Current:  NE: u2 = K2 * x + P2 * u1,  NE: u3 = K3 * x + P3 * u1
MM23[Block(4,6)] = -P2';   # η²⁻³ₜ₊₁

MM23[Block(4, 16)] = -B[:, player_control_list[1]]';  # λ²ₜ₊₁

# Row 5 corresponds to the gradient of the P2 Lagrangian with respect to u³ₜ₊₁.
MM23[Block(5,6)] = I(mⁱ); # η²⁻³ₜ₊₁

MM23[Block(5, 16)] = -B[:, player_control_list[3]]';  # λ²ₜ₊₁

# Row 6 corresponds to the gradient of the P3 Lagrangian with respect to u¹ₜ₊₁.
MM23[Block(6,7)] = I(mⁱ); # η³⁻¹ₜ₊₁

# TODO:  NE: u2 = -K2 * x - P2 * u1,  NE: u3 = -K3 * x - P3 * u1
# Current:  NE: u2 = K2 * x + P2 * u1,  NE: u3 = K3 * x + P3 * u1
MM23[Block(6,8)] = -P3';   # η³⁻²ₜ₊₁

MM23[Block(6, 17)] = -B[:, player_control_list[1]]';  # λ³ₜ₊₁

# Row 7 corresponds to the gradient of the P3 Lagrangian with respect to u²ₜ₊₁.
MM23[Block(7,8)] = I(mⁱ); # η³⁻²ₜ₊₁

MM23[Block(7, 17)] = -B[:, player_control_list[2]]';  # λ³ₜ₊₁

# Row 8 corresponds to the gradient of the P2 Lagrangian with respect to xₜ₊₁ (state x_T at time t=T-1).
MM23[Block(8, 3)] = I(n);     # λ²ₜ

# TODO:  NE: u2 = -K2 * x - P2 * u1,  NE: u3 = -K3 * x - P3 * u1
# Current:  NE: u2 = K2 * x + P2 * u1,  NE: u3 = K3 * x + P3 * u1
MM23[Block(8, 5)] = -K1';     # η²⁻¹ₜ₊₁
MM23[Block(8, 6)] = -K3';     # η²⁻³ₜ₊₁
MM23[Block(8, 9)] = Q[:,:,2]; # xₜ₊₁

MM23[Block(8, 16)] = -A';     # λ²ₜ₊₁

# Row 9 corresponds to the gradient of the P3 Lagrangian with respect to xₜ₊₁ (state x_T at time t=T-1).
MM23[Block(9, 4)] = I(n);     # λ³ₜ

# TODO:  NE: u2 = -K2 * x - P2 * u1,  NE: u3 = -K3 * x - P3 * u1
# Current:  NE: u2 = K2 * x + P2 * u1,  NE: u3 = K3 * x + P3 * u1
MM23[Block(9, 7)] = -K1';     # η³⁻¹ₜ₊₁
MM23[Block(9, 8)] = -K2';     # η³⁻²ₜ₊₁
MM23[Block(9, 9)] = Q[:,:,3]; # xₜ₊₁

MM23[Block(9, 17)] = -A';     # λ³ₜ₊₁

# Row 10 corresponds to the gradients of the P2 and P3 Lagrangian with respect to z¹ₜ₊₁ (i.e., KKT conditions at time t=T-1).
llast_block_col = length(zz_sizes²³ₜ);                              # Last column index for z¹ₜ
MM23[Block.(10:llast_block_col), Block(9)] = N1;                  # xₜ
MM23[Block.(10:llast_block_col),Block.(10:llast_block_col)] = M1  # zₜ₊₁


# 7. [skipped due to assumptions] Compute n23 block vector for P2 and P3 at penultimate stage t=T-1.

# 8. Compute the solution matrices for P2 and P3 at final stage t=T, i.e. for comparison.
ssol23 = -MM23 \ NN23; # P23 is the NE for players 2 and 3 for the terminal stage

KK2 = ssol23[Block(1,1)]; # KK2 is the feedback gain for player 2 at T-1
PP2 = ssol23[Block(1,2)]; # PP2 is the feedforward gain for player 2 at T-1
KK3 = ssol23[Block(2,1)]; # KK3 is the feedback gain for player 3 at T-1
PP3 = ssol23[Block(2,2)]; # PP3 is the feedforward gain for player 3 at T-1
# TODO:  NE: u2 = -K2 * x - P2 * u1,  NE: u3 = -K3 * x - P3 * u1
# Current:  NE: u2 = K2 * x + P2 * u1,  NE: u3 = K3 * x + P3 * u1


"""
Solve the Stackelberg hierarchy for P1 at the penultimate stage (t=T-1).
"""
# 1. We first define the ordering \mathcal{Z}¹ₜ of the state z¹ₜ at time t=T-1 for P1.
# \mathcal{Z}¹ₜ = [ u¹ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ, z²³ₜ ].
# \mathcal{Z}¹ₜ = [ u¹ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ, η¹⁻²ₜ₊₁, η¹⁻³ₜ₊₁, z²³ₜ ].

# 2. Compute the size of the stage state zⁱₜ for players 2 and 3 at time t=T-1.
# (2) + (2) + (2+2) + (2+2) + (36) = 48
zz_sizes¹ₜ = vcat([mⁱ, n, mⁱ, mⁱ, mⁱ, mⁱ], zz_sizes²³ₜ);
ss¹ₜ = sum(zz_sizes¹ₜ);

# TODO: This is consistent between stages, so can be done once at the beginning. Refactor to make it so.
# 3. Identify the ordering \mathcal{Y}¹ₜ of the information vector (i.e. state and leader information that P1
#    uses for decision-making) at t=T-1.
# \mathcal{Y}¹ₜ = [ xₜ ].

# TODO: This is consistent between stages, so can be done once at the beginning. No need to redefine variables.
# 4. Compute the size s̃¹ₜ of the information vector yⁱₜ for P1 at time t=T-1.
# (2) + (2) = 4 <=> 1 full state, 1 player control
ww_sizes¹ₜ = [n];
s̃s¹ₜ = sum(ww_sizes¹ₜ);

# 5. Compute the MM23 matrix (36x36) for P2 and P3 at penultimate stage t=T-1.
MM1 = BlockArray(zeros(ss¹ₜ, ss¹ₜ), zz_sizes¹ₜ, zz_sizes¹ₜ);

# 6. In addition, we set up the NN23 matrix (36x4) for P2 and P3 at penultimate stage t=T-1.
NN1 = BlockArray(zeros(ss¹ₜ, s̃s¹ₜ), zz_sizes¹ₜ, ww_sizes¹ₜ);

# # TODO: Change these to use the standard control law u = -Kx - Pu
# First block row of KKT conditions - gradient of P1's Lagrangian with respect to u¹ₜ.
MM1[Block(1,1)] = R[:,:,1]; # u¹ₜ
MM1[Block(1,2)] = -B1';     # λ¹ₜ
MM1[Block(1,3)] = -PP2';    # ψ¹⁻²
MM1[Block(1,4)] = -PP3';    # ψ¹⁻³

# Second block row of KKT conditions - gradient of P1's Lagrangian with respect to u²ₜ.
MM1[Block(2,2)] = -B2';
MM1[Block(2,3)] = I(mⁱ);

# Third block row of KKT conditions - gradient of P1's Lagrangian with respect to u³ₜ.
MM1[Block(3,2)] = -B3';
MM1[Block(3,4)] = I(mⁱ);

# Fourth block row of KKT conditions - gradient of P1's Lagrangian with respect to u²ₜ₊₁.
MM1[Block(4,5)]  = I(mⁱ);
MM1[Block(4,17)] = -B2';

# Fifth block row of KKT conditions - gradient of P1's Lagrangian with respect to u³ₜ₊₁.
MM1[Block(5,6)]  = I(mⁱ);

MM1[Block(5,17)] = -B3';

# Sixth block row of KKT conditions - gradient of P1's Lagrangian with respect to xₜ₊₁.
MM1[Block(6,2)] = I(n);
MM1[Block(6,5)] = -KK2';
MM1[Block(6,6)] = -KK3';

MM1[Block(6,15)] = Q[:, :, 1];
MM1[Block(6,17)] = -A';

# Seventh block row of KKT conditions - gradient of P1's Lagrangian with respect to z²³ₜ.
llast_block_col = length(zz_sizes¹ₜ)
MM1[Block.(7:llast_block_col), Block(1)] = NN23[Block.(1:18), Block(2)];    # Player 2's control input
MM1[Block.(7:llast_block_col), Block.(7:llast_block_col)] = MM23             # KKT conditions for P2 and P3

# Note: dynamics constraint exists in the KKT conditions for P2 and P3, so we only need to add this term to the existing dynamics row.
NN1[Block(9,1)] = -A; # xₜ

# 7. [skipped due to assumptions] Compute n23 block vector for P2 and P3 at final stage t=T.

# 8. Compute the solution matrices for P1 at final stage t=T, i.e. for comparison.
ssol1 = -MM1 \ NN1; # P1 is the NE for players 1, 2, and 3 for the terminal stage
# println("second with blocks: ", sol23)

KK1 = ssol1[Block(1,1)]; # K1 is the feedback gain for player 1.
# TODO: NE: u1 = -K1 * x
# Current: NE: u1 = K1 * x
