"""
This solver manually computes the Stackelberg hierarchy equilibrium for a 3-player LQ game, by appending each players'
KKT conditions at each time to the higher-level players' KKT conditions and at earlier times.
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
# \mathcal{Z}²ₜ = \mathcal{Z}³ₜ = [ u²ₜ, u³ₜ, xₜ₊₁, λ²ₜ, λ³ₜ ].

# 2. Compute the size of the stage state zⁱₜ for players 2 and 3 at time t=T.
# (2 + 2) + (6) + (6) + (6) = 20 <=> 2 player controls, 1 full state, 2 dynamics constraints
z_sizes²ₜ = z_sizes³ₜ = [mⁱ, mⁱ, n, n, n];
s²ₜ = s³ₜ = sum(z_sizes²ₜ);

# 3. Identify the ordering \mathcal{Y}ⁱₜ of the information vector (i.e. state and leader information that P2 and P3
#    use for decision-making) at t=T.
# \mathcal{Y}²ₜ = \mathcal{Y}³ₜ = [ xₜ, u¹ₜ ].

# 4. Compute the size s̃ⁱₜ of the information vector yⁱₜ for players 2 and 3 at time t=T.
# (6) + (2) = 8 <=> 1 full state, 1 player control
w_sizes²ₜ = w_sizes³ₜ = [n, mⁱ];
s̃²ₜ = s̃³ₜ = sum(w_sizes²ₜ);

# 5. Compute the M matrix for P2 and P3 at final stage t=T.
M23 = zeros(s²ₜ, s²ₜ);

Zeros2 = zeros(2, 2)
Zeros6 = zeros(6, 6)
Zeros6x2 = zeros(n, mⁱ)
Zeros2x6 = Zeros6x2'
I6 = I(n)
M23 = BlockArray([
    [R[:,:,2] Zeros2 Zeros2x6 -B[:,:,2]' Zeros2x6];
    [Zeros2 R[:,:,3] Zeros2x6 Zeros2x6 -B[:,:,3]'];
    [Zeros6x2 Zeros6x2 Q[:,:,2] I6 Zeros6];
    [Zeros6x2 Zeros6x2 Q[:,:,3] Zeros6 I6];
    [-B[:,:,2] -B[:,:,3] I6 Zeros6 Zeros6]
], z_sizes²ₜ, z_sizes²ₜ)

# 6. Compute the N matrix for P2 and P3 at final stage t=T.
N23 = BlockArray([
    zeros(s²ₜ - n, s̃²ₜ); # First 8 rows are zero
    -A -B[:,:,1] # u1
], z_sizes²ₜ, w_sizes²ₜ)

# 7. [skipped due to assumptions] Compute n block vector for P2 and P3 at final stage t=T.

# 8. Compute the solution matrices for P2 and P3 at final stage t=T, i.e. for comparison.
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
# For stage t=T, we require primal variables for the controls of all players (u¹ₜ, u²ₜ, u³ₜ) and the final state xₜ₊₁.
# We also require dual variables λ¹ₜ, λ²ₜ, and λ³ₜ for the dynamics constraints of each player, and 
# ψ¹⁻²ₜ, ψ¹⁻³ₜ for the policy constraints which tie P1's problem at time t=T to that of followers P2 and P3.

# 1. We first define the ordering \mathcal{Z}¹ₜ of the state z¹ₜ at time t=T for player 1.
# \mathcal{Z}¹ₜ = [ u¹ₜ, u²ₜ, u³ₜ, xₜ₊₁, λ²ₜ, λ³ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ ] = [ u¹ₜ, z²ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ ].
# We note that the ordering of z¹ₜ is a concatenation of the control of P1, the full state z²ₜ of P2 and P3, and
# the dual variables λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ.

# 2. Compute the size of the stage state z¹ₜ for player 1 at time t=T.
# (2 + 2 + 2) + (6) + (6 + 6) + (6) + (2 + 2) = 34 <=> 3 player controls, 1 full state, 3 dynamics constraints, 2 policy constraints
z_sizes¹ₜ = [mⁱ, mⁱ, mⁱ, n, n, n, n, mⁱ, mⁱ];
s¹ₜ = sum(z_sizes¹ₜ);

# 3. Identify the ordering \mathcal{Y}¹ₜ of the information vector (i.e. state and leader information that P1
#    use for decision-making) at t=T.
# \mathcal{Y}¹ₜ = [ xₜ ].

# 4. Compute the size s̃¹ₜ of the information vector yⁱₜ for player 1 at time t=T.
# (6) = 6 <=> 1 full state
w_sizes¹ₜ = [n];
s̃¹ₜ = sum(w_sizes¹ₜ);

# 5. Compute the M matrix for P1 at final stage t=T.
M1 = zeros(s¹ₜ, s¹ₜ);

Zeros2 = zeros(2, 2)
Zeros6 = zeros(6, 6)
Zeros6x2 = zeros(n, mⁱ)
Zeros2x6 = Zeros6x2'
I6 = I(n)
M1 = BlockArray([
    [R[:,:,1] zeros(mⁱ, s²ₜ)
    Zeros2 Zeros2x6 -B[:,:,2]' Zeros2x6];
    [Zeros2 R[:,:,3] Zeros2x6 Zeros2x6 -B[:,:,3]'];
    [Zeros6x2 Zeros6x2 Q[:,:,2] I6 Zeros6];
    [Zeros6x2 Zeros6x2 Q[:,:,3] Zeros6 I6];
    [-B[:,:,2] -B[:,:,3] I6 Zeros6 Zeros6]
], z_sizes¹ₜ, z_sizes¹ₜ)

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
