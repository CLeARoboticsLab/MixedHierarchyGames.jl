"""
This solver manually computes the Stackelberg hierarchy equilibrium for a 3-player LQ game, by appending each players'
KKT conditions at each time to the higher-level players' KKT conditions and at earlier times. We implement this by\
deriving the KKT conditions for one player at a time.

Note: We use modified (**) ordering mathcal{Z}ⁱₜ based on page 4 of Jingqi's goodnotes, which is different from the
      ordering used in block_kkt_solver.jl.
For a given node in the leadership hierarchy containing KKT conditions for players {a, b, c, ...}
mathcal{Z}ⁱₜ = [player i primal at t *(self)*;
                player i dual (*dynamics, then leadership, then policy*) at t;
                *follower primals at same time t*;
                *Nash primals (Nash at t) at same time*;
                future player KKT states at same time t, zⁱₜ;
                *shared variable (i.e. state which requires gradients wrt to it from multiple player Lagrangians)*,
                *future states zₜₜₜₜ* ]
Question: What do we do when we have multiple players in a more complex hierarchy?

Note: For the information vector, we use the ordering
mathcal{Y}ⁱₜ = [xₜ, u¹ₜ] for followers P2 and P3, and mathcal{Y}¹ₜ = [xₜ] for leader P1.
"""

# Import the game costs and dynamics from the problem definition.
include("problem.jl")

"""
We identify and solve the KKT conditions for all three players at final stage t=T, one player at a time. We must start
with the leaves of the Stackelberg hierarchy, which are the followers P2 and P3, and then work our way up to leader P1.
We arbitrarily choose to solve P2 first, then P3, and finally P1.
"""

"""
0. We first define orderings that don't change between time stages, i.e. for the information vector and the state for each player.
"""
# 0a. Identify the ordering \mathcal{Y}ⁱₜ of the information vector (i.e. state and leader information that Pi use for
#     decision-making). This ordering is dependent only on player and not on stage (as we assume leadership structure
#     is fixed).
# Note: We use least common subset because the info vector for each player contains state xₜ and any common leader
# \mathcal{Y}² = [ xₜ, u¹ₜ ].
# \mathcal{Y}³ = [ xₜ, u¹ₜ ].
# TODO: Confirm that least common subset is correct in more complicated scenarios.
# \mathcal{Y}²³ = (least common subset of involved players) [ xₜ, u¹ₜ ].
# \mathcal{Y}¹ = [ xₜ ].
# \mathcal{Y} = [ xₜ ].

# 0b. Compute the size s̃ⁱₜ of the information vector yⁱₜ for all players at each stage based on leadership relations.
# (2) + (2) = 4 <=> 1 full state, 1 player control (P1 is leader of P2 and P3)
w_sizes² = [n, mⁱ];
w_sizes³ = [n, mⁱ];
w_sizes²³ = [n, mⁱ];

# (2) = 2 <=> 1 full state
w_sizes¹ = [n];

s̃²ₜ = sum(w_sizes²);
s̃³ₜ = sum(w_sizes³);
s̃²³ₜ = sum(w_sizes²³);
s̃¹ₜ = sum(w_sizes¹);

# 0c. Within each stage, the order of computation/merging will be:
"""
M2, N2 -> K2, P2
M3, N3 -> K3, P2
M23, N23 -> K2, K3, P2, P3
M1, N1 -> K1, P1
"""

# 0d. The ordering, \mathcal{Z}ⁱₜ, of the state zⁱₜ within a given stage should be the same for all players, except for
#     the dependence on the future states. 
#     Note: the future KKT condition sizes will vary, but we can set an ordering for the KKT conditions for each player.
"""
mathcal{Z}ⁱₜ = [player i primals at t *(self, then follower, then Nash)*;
                player i dual (dynamics + policy) at t;
                *follower KKT conditions at t*;
                *Nash KKT conditions at t*;
                future player KKT states at t, zⁱₜ;
                *shared variable (i.e. state)*]

Note: h ∈ t + {1, ..., T - t} = {t+1, ... , T}.

Final stage:
P2: mathcal{Z}²ₜ = [ u²ₜ, λ²ₜ, u³ₜ, xₜ₊₁ ].
P3: mathcal{Z}³ₜ = [ u³ₜ, λ³ₜ, u²ₜ, xₜ₊₁ ].
P23: mathcal{Z}²³ₜ = [ u²ₜ, u³ₜ, λ²ₜ, λ³ₜ, xₜ₊₁ ].
P1: mathcal{Z}¹ₜ = [ u¹ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ, z²³ₜ ].

All stages (t < T & t = T):
Note: ηₜ₊₁ = {}, zⁱₜ₊₁ = {}
Q: Does z¹ₜ₊₁ include all {η} at t in {t+2, ..., T}? If so, we can adjust the current time ηs. Adjusted per this comment.
P2: mathcal{Z}²ₜ   -> z²ₜ  = [ u²ₜ, λ²ₜ, η²⁻¹ₜ₊₁, η²⁻³ₜ₊₁, u³ₜ, xₜ₊₁, z¹ₜ₊₁ ].
P3: mathcal{Z}³ₜ   -> z³ₜ  = [ u³ₜ, λ³ₜ, η³⁻¹ₜ₊₁, η³⁻²ₜ₊₁, u²ₜ, xₜ₊₁, z¹ₜ₊₁ ].
P23: mathcal{Z}²³ₜ -> z²³ₜ = [ u²ₜ, u³ₜ, λ²ₜ, λ³ₜ, η²⁻¹ₜ₊₁, η²⁻³ₜ₊₁, η³⁻¹ₜ₊₁, η³⁻²ₜ₊₁, xₜ₊₁, z¹ₜ₊₁ ].
P1: mathcal{Z}¹ₜ   -> z¹ₜ  = [ u¹ₜ, λ¹ₜ, ψ¹⁻²ₜ, ψ¹⁻³ₜ, z²³ₜ ].
"""


# 0e. The ordering of the KKT conditions within a given stage should be the same for all players. Note: the future KKT
#     condition sizes will vary, but we can set an ordering for the KKT conditions for each player.

"""
P2:
gradient of P2 wrt u²ₜ
gradient of P2 wrt λ²ₜ
gradient of P2 wrt u³ₜ
gradient of P2 wrt xₜ₊₁
"""


"""
P3:
gradient of P3 wrt u³ₜ
gradient of P3 wrt λ³ₜ
gradient of P3 wrt u²ₜ
gradient of P3 wrt xₜ₊₁
"""


"""
P23:
gradient of P2 wrt u²ₜ
gradient of P3 wrt u³ₜ
gradient of P2 wrt λ²ₜ (combined with following since dynamics constraints are shared)
gradient of P3 wrt λ³ₜ (removed because it is a duplicate)
gradient of P2 wrt xₜ₊₁
gradient of P3 wrt xₜ₊₁
"""

"""
Solve the Stackelberg hierarchy for the follower P2 at the terminal stage.
"""
# For stage t=T, we require primal variables for the controls of P2 (u²ₜ) and players which are followers (none) or
# related through a Nash relationship (u³ₜ), as well as the final state xₜ₊₁. The state z²ₜ also contains a dual
# variable (λ²ₜ) for P2's dynamics constraints.

# 1. We first define the ordering \mathcal{Z}²ₜ of the state z²ₜ at time t=T for P2.
# TODO: As λ³ₜ does not show up in the KKT conditions for P2, we do not include it in the state, as it will be merged.
# Q: Do we need to include associated duals for P3 in the state for P2 when we must have the control for P3?
# Q: Is each constraint for Pi known to other players P(-i)? If so, we must include the duals for P3 in the state.
# ASSUME: We assume all constraints are private to each player, so we do not include the duals for P3 in the state for P2.
# \mathcal{Z}²ₜ = [ u²ₜ, λ²ₜ, u³ₜ, xₜ₊₁ ].

# 2. Compute the size of the stage state z²ₜ for P2 at time t=T.
# (2 + 2) + (2) + (2) = 8 <=> 2 player controls, 1 dynamics constraint, 1 full state
z_sizes²ₜ = [mⁱ, mⁱ, n, n];
s²ₜ = sum(z_sizes²ₜ);

# 3. Construct M2 as a 8x8 block matrix, N2 as a 8x4 matrix.
M2 = BlockArray(zeros(s²ₜ, s²ₜ), z_sizes²ₜ, z_sizes²ₜ);
N2 = BlockArray(zeros(s²ₜ, s̃²ₜ), z_sizes²ₜ, w_sizes²);

# Row 1 is the gradient of P2's Lagrangian with respect to u²ₜ.
M2[Block(1,1)] = R[:,:,2]; # u²ₜ
M2[Block(1,2)] = -B2';     # λ²ₜ

# Row 2 is the gradient of P2's Lagrangian with respect to λ²ₜ (i.e. dynamics).
M2[Block(2,1)] = -B2;      # u²ₜ
M2[Block(2,3)] = -B3;      # u³ₜ
M2[Block(2,4)] = I(n);     # xₜ₊₁

N2[Block(2,1)] = -A;       # xₜ₊₁
N2[Block(2,2)] = -B1;      # u¹ₜ

# Row 3 is the gradient of P2's Lagrangian with respect to u³ₜ.
M2[Block(3,2)] = -B3';     # λ²ₜ

# Row 4 is the gradient of P2's Lagrangian with respect to xₜ₊₁.
M2[Block(4,1)] = I(n);     # u²ₜ
M2[Block(4,4)] = Q[:,:,2]; # xₜ₊₁


# 4a. Compute the size of the stage state z³ₜ for P3 at time t=T.
# (2 + 2) + (2) + (2) = 8 <=> 2 player controls, 1 dynamics constraint, 1 full state
z_sizes³ₜ = [mⁱ, mⁱ, n, n];
s³ₜ = sum(z_sizes³ₜ);

# 4b. Construct M3 as a 8x8 block matrix, N3 as a 8x4 matrix.
# \mathcal{Z}³ₜ = [ u³ₜ, λ³ₜ, u²ₜ, xₜ₊₁ ]. TODO: All caveats for P2 apply here.
M3 = BlockArray(zeros(s³ₜ, s³ₜ), z_sizes³ₜ, z_sizes³ₜ);
N3 = BlockArray(zeros(s³ₜ, s̃³ₜ), z_sizes³ₜ, w_sizes³);

# Row 1 is the gradient of P3's Lagrangian with respect to u³ₜ.
M3[Block(1,1)] = R[:,:,3]; # u³ₜ
M3[Block(1,2)] = -B3';     # λ³ₜ

# Row 2 is the gradient of P3's Lagrangian with respect to λ³ₜ (i.e. dynamics).
M3[Block(2,1)] = -B3;      # u³ₜ
M3[Block(2,3)] = -B2;      # u²ₜ
M3[Block(2,4)] = I(n);     # xₜ₊₁

N3[Block(2,1)] = -A;       # xₜ₊₁
N3[Block(2,2)] = -B1;      # u¹ₜ

# Row 3 is the gradient of P3's Lagrangian with respect to u²ₜ.
M3[Block(3,2)] = -B2';     # λ³ₜ

# Row 4 is the gradient of P3's Lagrangian with respect to xₜ₊₁.
M3[Block(4,1)] = I(n);     # u³ₜ
M3[Block(4,4)] = Q[:,:,3]; # xₜ₊₁


# 5a. Merge M2 and M3. Construct M23 as a 10x10 block matrix, N23 as a 10x4 matrix. 
# \mathcal{Z}²³ₜ = [ u²ₜ, u³ₜ, λ²ₜ, λ³ₜ, xₜ₊₁ ].
# P23 row order:
# gradient of P2 wrt u²ₜ
# gradient of P3 wrt u³ₜ
# gradient of P2 wrt λ²ₜ (combined with following since dynamics constraints are shared)
# gradient of P3 wrt λ³ₜ (removed because it is a duplicate)
# gradient of P2 wrt xₜ₊₁
# gradient of P3 wrt xₜ₊₁

# 5b. Compute the size of the stage state zⁱₜ for players 2 and 3 at time t=T.
# (2 + 2) + (2) + (2) + (2) = 10 <=> 2 player controls, 2 dynamics constraints, 1 full state
z_sizes²³ₜ = [mⁱ, mⁱ, n, n, n];
s²³ₜ = sum(z_sizes²³ₜ);

# 5c. Build M and N based on previous matrices M2, N2, M3, and N3.
M23 = BlockArray(zeros(s²³ₜ, s²³ₜ), z_sizes²³ₜ, z_sizes²³ₜ);
N23 = BlockArray(zeros(s²³ₜ, s̃²³ₜ), z_sizes²³ₜ, w_sizes²³);

# First, identify the primal variables for the players being merged (P2 and P3 have controls u2 and u3).
# Row 1 is the gradient of P2's Lagrangian with respect to u²ₜ.
M23[Block(1,1)] = M2[Block(1,1)]; # u²ₜ
M23[Block(1,3)] = M2[Block(1,2)]; # λ²ₜ

# Row 2 is the gradient of P3's Lagrangian with respect to u³ₜ.
M23[Block(2,2)] = M3[Block(1,1)]; # u³ₜ
M23[Block(2,4)] = M3[Block(1,2)]; # λ³ₜ

# Row 3 is the shared constraint, i.e. dynamics, i.e. gradient of Pi's Lagrangian with respect to uⁱₜ.
M23[Block(3,1)] = M2[Block(2,1)]; # u²ₜ
M23[Block(3,2)] = M3[Block(2,1)]; # u³ₜ
M23[Block(3,5)] = M2[Block(2,4)]; # xₜ₊₁

N23[Block(3,1)] = N3[Block(2,1)]; # xₜ₊₁
N23[Block(3,2)] = N3[Block(2,2)]; # u¹ₜ

# Row 4 is the gradient of P2's Lagrangian with respect to xₜ₊₁.
M23[Block(4,3)] = M2[Block(4,1)]; # λ²ₜ
M23[Block(4,5)] = M2[Block(4,4)]; # xₜ₊₁

# Row 5 is the gradient of P3's Lagrangian with respect to xₜ₊₁.
M23[Block(5,4)] = M3[Block(4,1)]; # λ³ₜ
M23[Block(5,5)] = M3[Block(4,4)]; # xₜ₊₁


