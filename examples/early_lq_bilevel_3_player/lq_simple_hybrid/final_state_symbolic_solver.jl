include("problem.jl")

"""Solve the KKT conditions for the Stackelberg hierarchy problem.
"""

"""
Note: For the information vector, we use the ordering
mathcal{Y}ⁱₜ = [xₜ] for P1 and P2, and mathcal{Y}³ₜ = [xₜ, u²ₜ] for follower P3.
"""
function get_roots(graph::SimpleDiGraph)
    # Find all vertices with no incoming edges (roots).
    return [v for v in vertices(graph) if indegree(graph, v) == 0]
end

function get_all_leaders(graph::SimpleDiGraph, node::Int)
    parents_path = []

    parents = inneighbors(graph, node)

    while !isempty(parents)
        # Identify and save each parent until we reach the root.
        # TODO: We assume this is a tree for now.
        parent = only(parents)
        push!(parents_path, parent)

        # Get next grandparent.
        parents = inneighbors(graph, parent)
    end

    return parents_path
end

# Get every follower of a node in the graph in an arbitrary order.
# TODO: Make this order more educated once we have a better understanding of the hierarchy.
# Note: Assumes that every vertex is the root of a tree.
function get_all_followers(graph::SimpleDiGraph, node)
    all_children = outneighbors(graph, node)

    children = all_children
    has_next_layer = !isempty(children)
    while has_next_layer
        grandchildren = []
        for child in children
            # Get all children of the current child.
            # Note: This is a breadth-first search.
            for grandchild in outneighbors(graph, child)
                push!(all_children, grandchild)
                push!(grandchildren, grandchild)
            end
        end
        children = grandchildren
        has_next_layer = !isempty(children)
    end
    return all_children
end

# Note: Assumes that every vertex is the root of a tree.
function compute_info_vector_ordering(graph::SimpleDiGraph)
    roots = get_roots(graph)
    # TODO: Fix the NaN for variable length tuples. Use make_string/make_symbol?
    info_vector_variables = Dict(root => [:x] for root in roots)

    parents = roots
    has_next_layer = true
    while has_next_layer
        next_layer_parents = []
        for parent in parents
            children = outneighbors(graph, parent)
            for child in children
                # The child's info vector is the parent's, with the parent control appended.
                child_info_vector = vcat(info_vector_variables[parent], [Symbol(:u, parent)])
                info_vector_variables[child] = child_info_vector

                # Add each children of current parents to next layer nodes vector.
                append!(next_layer_parents, child)
            end
        end
        parents = next_layer_parents
        has_next_layer = !isempty(parents)
    end
    return info_vector_variables
end

# # TODO: Generalize this to make it work over multiple horizons.
# struct ZOrdering{T1, T2, T3, T4, T5, T6, T7}
#     self_primals::T1
#     self_duals::T2
#     follower_primals::T3
#     follower_duals::T4
#     # TODO: Remove if not needed.
#     nash_primals::T5
#     next_state::T6
#     future_z::T7
# end
    
# # TODO: Does not current have future policy duals.
# function compute_z_ordering_for_player(graph::SimpleDiGraph, node)
# TODO: Implement this.
# end

"""
We identify and solve the KKT conditions for all three players at final stage t=T, one player at a time. We must start
with the leaves of the Stackelberg hierarchy, which are the players P1 and P3, and then work our way for each one.
We arbitrarily choose to set up KKT conditions for P1 first, then P3, and finally P2.
"""

"""
0. We first define orderings that don't change between time stages, i.e. for the information vector and the state for each player.
"""
Ys = compute_info_vector_ordering(g)
@assert Ys[1] == Ys[2] "P1 and P2 should have same info vector." # P1 and P2 have the same info vector.
# Ys[(1,2)] = Ys[1]
Y_sizes = Dict(k => lookup_varsize.(v) for (k, v) in Ys)


# TODO: Come up with a function to programmatically compute z-orderings for each player.
Zs = Dict(
        1 => [:u1, :λ1, :u2,  :u3, :xtp1], # P1
        3 => [:u3, :λ3, :u1,  :xtp1], # P3
        2 => [:u2, :λ2, :η23, :u3, :λ3, :u1, :xtp1],  # P2
        # (1,2) => [:u1, :λ1, :u2, :λ2, :η23, :u3, :λ3, :xtp1] # P1 + P2 (all subtrees of hierarchy)
    )
Z_sizes = Dict(k => lookup_varsize.(v) for (k, v) in Zs)

# TODO: Come up with a function to progammatically order KKT conditions for each player.
# (i, :x) means the derivative of Lagrangian Lⁱ wrt x.
# (0, :f) indicates the constraint named f, which is the dynamics constraint.

KKT_orderings = Dict(
    1 => [(1, :u1),
          (1, :u2), # Nash constraint is probably an extra KKT constraint
          (1, :u3), # Nash constraint is probably an extra KKT constraint
          (0, :f), 
          (1, :xtp1)], # P1
    3 => [(3, :u3),
          (3, :u1), # Nash constraint is probably an extra KKT constraint
          (0, :f),
          (3, :xtp1)], # P3
    2 => [(2, :u2),
          (2, :u3),
          (2, :u1), # Nash constraint is probably an extra KKT constraint
          (0, :f),
          (2, :xtp1),
          (3, :xtp1)],  # P2
    (1,2) => [(1, :u1),
              (2, :u2),
              (2, :u3),
              (0, :f),
              (1, :xtp1),
              (2, :xtp1),
              (3, :xtp1),
              (0, :π)] # P1 + P2
)


# Set up P1 KKT conditions (leaf).
z1_len = sum(Z_sizes[1])
y1_len = sum(Y_sizes[1]);
M1 = BlockArray(zeros(z1_len, z1_len), Z_sizes[1], Z_sizes[1]);
N1 = BlockArray(zeros(z1_len, y1_len), Z_sizes[1], Y_sizes[1]);

# Row 1: ∇L¹ wrt u¹
M1[Block(1), Block(1)] = R[:, :, 1];   # u¹
M1[Block(1), Block(2)] = -B1'; # λ¹

# Row 2: ∇L¹ wrt u² - Nash (may need to remove)
M1[Block(2), Block(2)] = -B2'; # λ¹

# Row 3: ∇L¹ wrt u³ - Nash (may need to remove)
M1[Block(3), Block(2)] = -B3'; # λ¹

# Row 4: dynamics constraint
M1[Block(4), Block(1)] = -B1;  # u¹
M1[Block(4), Block(3)] = -B2;  # u³
M1[Block(4), Block(4)] = -B3;  # u²
M1[Block(4), Block(5)] = I(2); # xtp1

N1[Block(4), Block(1)] = -A;   # xt

# Row 5: ∇L¹ wrt xtp1
M1[Block(5), Block(2)] = I(2);        # λ¹
M1[Block(5), Block(5)] = Q[:,:,1]; # xtp1

# Large solution - probably not right because it has extra KKT conditions.
S1 = M1 \ N1; # Solve the KKT conditions for P1
K1 = S1[Block(1), Block(1)] # K1 is the feedback gain for player 1
P1 = zeros(2,2) # P1 is the P2 control gain for player 1

# Small solution
MM1 = M1[Block.([1,4,5]), Block.([1,2,5])]
NN1 = N1[Block.([1,4,5]), :]
SS1 = -MM1 \ NN1;             # Solve the KKT conditions for P1
KK1 = SS1[Block(1), Block(1)] # K1 is the feedback gain for player 1
PP1 = P1                      # No leadership information besides state


# Set up P3 KKT conditions (leaf).
z3_len = sum(Z_sizes[3])
y3_len = sum(Y_sizes[3]);
M3 = BlockArray(zeros(z3_len, z3_len), Z_sizes[3], Z_sizes[3]);
N3 = BlockArray(zeros(z3_len, y3_len), Z_sizes[3], Y_sizes[3]);

# Row 1: ∇L³ wrt u³
M3[Block(1), Block(1)] = R[:, :, 3];   # u³
M3[Block(1), Block(2)] = -B3'; # λ³

# Row 2: ∇L³ wrt u¹ - Nash (may need to remove)
M3[Block(2), Block(2)] = -B1'; # λ³

# Row 3: dynamics constraint
M3[Block(3), Block(1)] = -B3; # u³
M3[Block(3), Block(3)] = -B1; # u¹
M3[Block(3), Block(4)] = I(2);

N3[Block(3), Block(1)] = -A;
N3[Block(3), Block(2)] = -B2; # u²

# Row 4: ∇L³ wrt xtp1
M3[Block(4), Block(2)] = I(2);        # λ³
M3[Block(4), Block(4)] = Q[:,:,3]; # xtp1

# Large solution - probably not right because it has extra KKT conditions.
S3 = M3 \ N3; # Solve the KKT conditions for P3
K3 = S3[Block(1), Block(1)] # K3 is the feedback gain for player 3
P3 = S3[Block(1), Block(2)] # P3 is the P2 control gain for player 3

# Small solution
MM3 = M3[Block.([1,3,4]), Block.([1,2,4])]
NN3 = N3[Block.([1,3,4]), :]
SS3 = -MM3 \ NN3; # Solve the KKT conditions for P3
KK3 = SS3[Block(1), Block(1)] # K3 is the feedback gain for player 3
PP3 = SS3[Block(1), Block(2)] # P3 is the P2 control gain for player 3


# Set up P2 KKT conditions (Stackelberg merge with P3).
z2_len = sum(Z_sizes[2])
y2_len = sum(Y_sizes[2]);
M2 = BlockArray(zeros(z2_len, z2_len), Z_sizes[2], Z_sizes[2]);
N2 = BlockArray(zeros(z2_len, y2_len), Z_sizes[2], Y_sizes[2]);

# Row 1: ∇L² wrt u²
M2[Block(1), Block(1)] = R[:, :, 2]; # u²
M2[Block(1), Block(2)] = -B2';       # λ²
M2[Block(1), Block(3)] = PP3;        # η²³

# Row 2: ∇L² wrt u³
M2[Block(2), Block(2)] = -B2'; # λ²
M2[Block(2), Block(3)] = I(2); # η²³

# Row 3: ∇L² wrt u¹ - Nash (may need to remove)
M2[Block(3), Block(2)] = -B1'; # λ²

# Row 4: dynamics constraint
M2[Block(4), Block(1)] = -B2;  # u²
M2[Block(4), Block(4)] = -B3;  # u³
M2[Block(4), Block(6)] = -B1;  # u¹
M2[Block(4), Block(7)] = I(2); # xtp1

N2[Block(4), :] = -A;          # xt

# Row 5: ∇L² wrt xtp1
M2[Block(5), Block(2)] = I(2);     # λ²
M2[Block(5), Block(7)] = Q[:,:,2]; # xtp1

# Row 6: ∇L³ wrt xtp1
M2[Block(6), Block(5)] = I(2);     # λ³
M2[Block(6), Block(7)] = Q[:,:,3]; # xtp1

# Row 7: policy constraint
M2[Block(7), Block(1)] = PP3;  # u²
M2[Block(7), Block(4)] = I(2); # u³

N2[Block(7), Block(1)] = KK3;  # xt

# Large solution - probably not right because it has extra KKT conditions.
S2 = M2 \ N2; # Solve the KKT conditions for P2
K2 = S2[Block(1), Block(1)] # K2 is the feedback gain for player 2
P2 = zeros(2,2) # No control information provided to player 2

# Small solution
MM2 = M2[Block.([1,2,4,5,6,7]), Block.([1,2,3,4,5,7])]
NN2 = N2[Block.([1,2,4,5,6,7]), :]
SS2 = -MM2 \ NN2; # Solve the KKT conditions for P2
KK2 = SS2[Block(1), Block(1)] # K2 is the feedback gain for player 2
PP2 = zeros(2,2) # No control information provided to player 2


# Set up P1+P2 KKT conditions (Nash merge).
Y12_ordering = Ys[1]
Y12_sizes = lookup_varsize.(Y12_ordering)
Z12_ordering = [:u1, :λ1, :u2, :λ2, :η23, :u3, :λ3, :xtp1]
Z12_sizes = lookup_varsize.(Z12_ordering)
z12_len = sum(Z12_sizes)
y12_len = sum(Y12_sizes);
M12 = BlockArray(zeros(z12_len, z12_len), Z12_sizes, Z12_sizes);
N12 = BlockArray(zeros(z12_len, y12_len), Z12_sizes, Y12_sizes);

# Row 1: ∇L¹ wrt u¹
M12[Block(1), Block.(1:2)] = M1[Block(1), Block.(1:2)]

# Row 2: ∇L² wrt u²
# Row 3: ∇L² wrt u³
M12[Block.(2:3), Block.(3:5)] = M2[Block.(1:2), Block.(1:3)]

# Row 4: dynamics constraint
M12[Block(4), Block(1)] = -B1;  # u¹
M12[Block(4), Block(3)] = -B2;  # u²
M12[Block(4), Block(6)] = -B3;  # u³
M12[Block(4), Block(8)] = I(2); # xtp1

N12[Block(4), Block(1)] = -A;   # xt

# Row 5: ∇L¹ wrt xtp1
M12[Block(5), Block(2)] = I(2);     # λ¹
M12[Block(5), Block(8)] = Q[:,:,1]; # xtp1

# Row 6: ∇L² wrt xtp1
M12[Block(6), Block(4)] = I(2);     # λ²
M12[Block(6), Block(8)] = Q[:,:,2]; # xtp1

# Row 7: ∇L³ wrt xtp1
M12[Block(7), Block(7)] = I(2);     # λ³
M12[Block(7), Block(8)] = Q[:,:,3]; # xtp1

# Row 8: leader policy constraint π³
M12[Block(8), Block(3)] = PP3;  # u²
M12[Block(8), Block(6)] = I(2); # u³

N12[Block(8), Block(1)] = KK3;  # xt


# Solution for all variables, assuming PP3 and KK3 are valid.
S12 = -M12 \ N12; # Solve the KKT conditions for P2
# K2 = S12[Block(1), Block(1)] # K2 is the feedback gain for player 2
# P2 = zeros(2,2) # No control information provided to player 2

KKK1 = S12[Block(1), Block(1)] # KKK1 is the feedback gain for player 1
KKK2 = S12[Block(3), Block(1)] # KKK2 is the feedback gain for player 2
KKK3 = S12[Block(6), Block(1)] # KKK3 is the feedback gain for player 3
