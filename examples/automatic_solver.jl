using Graphs

# TODO: Turn this into an extension of the string type using my own type.
function make_symbol(args...)
    variable_name = args[1]
    time = last(args)

    num_items = length(args)

    @assert variable_name in [:x, :u, :λ, :ψ, :μ, :z, :M, :N, :Φ]
    variable_name_str = string(variable_name)

    if variable_name == :x && num_items == 2 # Just :x
        return Symbol(variable_name_str * "_" * string(time))
    elseif variable_name in [:u, :λ, :z] && num_items == 3
       return Symbol(variable_name_str * "^" * string(args[2]) * "_" * string(time))
    elseif variable_name in [:ψ, :μ] && num_items == 4
        return Symbol(variable_name_str * "^(" * string(args[2]) * "-" * string(args[3]) * ")" * "_" * string(time))
    elseif variable_name in [:z] && num_items > 3
        # For z variables, we assume the inputs are of the form (z, i, j, ..., t)
        indices = join(string.(args[2:num_items-1]), ",")
        return Symbol(variable_name_str * "^(" * indices * ")" * "_"* string(time))
    else
        error("Invalid format has number of args $(num_items) for $args.")
    end
end


#### GRAPH UTILITIES ####

function has_leader(graph::SimpleDiGraph, node::Int)
    return !is_root(graph, node)
end

function is_root(graph::SimpleDiGraph, node::Int)
    return iszero(indegree(graph, v))
    
end

function get_roots(graph::SimpleDiGraph)
    # Find all vertices with no incoming edges (roots).
    return [v for v in vertices(graph) if is_root(graph, v)]
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

# function compute_info_vector_ordering(graph::SimpleDiGraph)
#     roots = get_roots(graph)
#     # TODO: Fix the NaN for variable length tuples. Use make_string/make_symbol?
#     info_vector_variables = Dict(root => [:x] for root in roots)

#     parents = roots
#     has_next_layer = true
#     while has_next_layer
#         next_layer_parents = []
#         for parent in parents
#             children = outneighbors(graph, parent)
#             for child in children
#                 # The child's info vector is the parent's, with the parent control appended.
#                 child_info_vector = vcat(info_vector_variables[parent], [Symbol(:u, parent)])
#                 info_vector_variables[child] = child_info_vector

#                 # Add each children of current parents to next layer nodes vector.
#                 append!(next_layer_parents, child)
#             end
#         end
#         parents = next_layer_parents
#         has_next_layer = !isempty(parents)
#     end
#     return info_vector_variables
# end



###### UTILS FOR PATH SOLVER  ######
# TODO: Fix to make ti general based on whatever expressions are needed.
# function solve_with_path()
#     # Final MCP vector: leader stationarity + leader constraints + follower KKT
#     F = Vector{symbolic_type}([
#         # TODO: Fill in with actual expressions, automatically.
#     ])

#     # Main.@infiltrate

#     # variables = vcat(z₁, z₂, z₃, λ₁, λ₂, λ₃)
#     z̲ = fill(-Inf, length(F));
#     z̅ = fill(Inf, length(F))

#     # Solve via PATH
#     parameter_value = [1e-5]
#     parametric_mcp = ParametricMCPs.ParametricMCP(F, variables, [θ], z̲, z̅; compute_sensitivities = false)
#     z_sol, status, info = ParametricMCPs.solve(
#         parametric_mcp,
#         parameter_value;
#         initial_guess = zeros(length(variables)),
#         verbose = false,
#         cumulative_iteration_limit = 100000,
#         proximal_perturbation = 1e-2,
#         # major_iteration_limit = 1000,
#         # minor_iteration_limit = 2000,
#         # nms_initial_reference_factor = 50,
#         use_basics = true,
#         use_start = true,
#     )
#     @show status
# end

function is_leaf(graph::SimpleDiGraph, node::Int)
    return outdegree(graph, node) == 0
end

function get_lagrangians(G::SimpleDiGraph,
                         Js::Dict{Int, Any},
                         zs::Dict{Int, Any},
                         λs::Dict{Int, Any},
                         μs::Dict{Tuple{Int,Int}, Any},
                         gs::Dict{Int, Any}, 
                         ws::Dict{Int, Any},
                         ys::Dict{Int, Any},
                         θ)

    # Compute reverse topological order to construct lagrangians and KKT conditions from leaves to root.
    order = reverse(topological_sort(G))

    println("Topological order of vertices:")
    for ii in order
        println(v)
    end

    for ii in order
        # Include the objective of the player and its constraint term.
        Lᵢ = Js[ii](zs..., θ) - λs[ii]' * gs[ii](zs[ii])

        # If the current player has no followers, then we can compute its KKT conditions directly.
        if is_leaf(G, ii)
            πᵢ = vcat(Symbolics.gradient(Lᵢ, zs[ii]), # stationarity of follower only w.r.t its own vars
                                      gs[ii](zs[ii])) # constraints for current player

            # Compute ∇ᵢπᵢ = Mᵢ wᵢ + Nᵢ yᵢ = 0.
            # TODO: Save in a new
            Mᵢ = Symbolics.jacobian(πᵢ, ws[ii])
            Nᵢ = Symbolics.jacobian(πᵢ, ys[ii])
        end

        # If it has followers, then add the follower's stationarity conditions.
        if !is_leaf(G, ii)

            # Iterate in breadth-first order over the followers.
            for jj in BFSIterator(G, ii)

                # Skip the current player.
                if ii == jj
                    continue
                end

                # TODO: Look closely when I'm clear-eyed.

                πᵢ = vcat(πᵢ, Φʲ) # stationarity of follower j w.r.t its own vars




            followers = get_all_followers(G, order[1])
            for j in followers
                Φʲ
                Lᵢ -= μs[(ii, jj)]' * (zs[jj] - )
            end
        end

    end

    # roots = get_roots(graph)
    # # TODO: Fix the NaN for variable length tuples. Use make_string/make_symbol?
    # info_vector_variables = Dict(root => [:x] for root in roots)

    # parents = roots
    # has_next_layer = true
    # while has_next_layer
    #     next_layer_parents = []
    #     for parent in parents
    #         children = outneighbors(graph, parent)
    #         for child in children
    #             # The child's info vector is the parent's, with the parent control appended.
    #             child_info_vector = vcat(info_vector_variables[parent], [Symbol(:u, parent)])
    #             info_vector_variables[child] = child_info_vector

    #             # Add each children of current parents to next layer nodes vector.
    #             append!(next_layer_parents, child)
    #         end
    #     end
    #     parents = next_layer_parents
    #     has_next_layer = !isempty(parents)
    # end
    # return info_vector_variables

end


# Main body of algorithm implementation. Will restructure as needed.
function main()
    N = 3 # number of players

    # Set up the information structure.
    # This defines a stackelberg chain with three players, where P1 is the leader of P2, and P1+P2 are leaders of P3.
    G = SimpleDiGraph(N);
    add_edge!(G, 1, 2); # P1 -> P2
    add_edge!(G, 2, 3); # P2 -> P3


    H = 1
    Hp1 = H+1 # number of planning stages is 1 for OL game.

    # Helper function
    flatten(vs) = collect(Iterators.flatten(vs))

    # Initial sizing of various dimensions.
    N = 3 # number of players
    T = 10 # time horizon
    state_dimension = 2 # player 1,2,3 state dimension
    control_dimension = 2 # player 1,2,3 control dimension
    x_dim = state_dimension * (T+1)
    u_dim = control_dimension * (T+1)
    aggre_state_dimension = x_dim * N
    aggre_control_dimension = u_dim * N
    total_dimension = aggre_state_dimension + aggre_control_dimension
    primal_dimension_per_player = x_dim + u_dim




    # Dynamics
    Δt = 0.5 # time step
    # A = I(state_dimension * num_players)
    # B¹ = [Δt * I(control_dimension); zeros(4, 2)]
    # B² = [zeros(2, 2); Δt * I(control_dimension); zeros(2, 2)]
    # B³ = [zeros(4, 2); Δt * I(control_dimension)]
    # B = [B¹ B² B³]

    # Dynamics are the only constraints (for now).
    function dynamics(z, t)
        (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        x = xs[t]
        u = us[t]
        xp1 = xs[t+1]
        # rows 3:4 for p2 in A, and columns 3:4 for p2 in B when using the full stacked system
        # but since A is I and B is block-diagonal by design, you can just write:
        return xp1 - x - Δt*u
    end

    # Set up the equality constraints for each player.
    ics = [[-2.0; 2.0], 
           [ 0.5; 1.0], 
           [-1.0; 2.0]] # initial conditions for each player

    make_ic_constraint(i) = function (zᵢ)
        (; xs, us) = unflatten_trajectory(zᵢ, state_dimension, control_dimension)
        x1 = xs[1]
        return x1 - ics[i]
    end

    dynamics_constraint(zᵢ) = mapreduce(vcat, 1:T) do t
            dynamics(zᵢ, t)
        end

    gs = [function (zᵢ) vcat(dynamics_constraint(zᵢ), 
                             make_ic_constraint(i)(zᵢ))
          end for i in 1:N] # each player has the same dynamics constraint



    # Construct symbols for each player's decision variables.
    # TODO: Construct sizes and orderings.
    backend = SymbolicTracingUtils.SymbolicsBackend()
    zs = [SymbolicTracingUtils.make_variables(
            backend,
            make_symbol(:z, i, H),
            primal_dimension_per_player,
        ) for i in 1:N]

    λs = [SymbolicTracingUtils.make_variables(
            backend,
            make_symbol(:λ, i, H),
            length(gs[i](zs[i]))
        ) for i in 1:N]

    μs = Dict{Tuple{Int,Int}, Any}()
    ws = Dict{Int, Any}()
    ys = Dict{Int, Any}()
    # Ms = Dict{Int, Any}()
    # Ns = Dict{Int, Any}()
    for i in 1:N
        # Get all followers of i, create the variable for each, and store them in a Dict.
        followers = get_all_followers(G, i)
        for j in followers
            μs[(i,j)] = SymbolicTracingUtils.make_variables(
                backend,
                make_symbol(:μ, i, j, H),
                primal_dimension_per_player
            )
        end

        # yᵢ is the information vector containing states zᴸ associated with leaders L of i.
        leaders = get_all_leaders(G, i)
        ys[i] = vcat(zs[leaders])

        # wᵢ is used to identify policy constraints by leaders of i.
        # Construct wᵢ by adding (1) zs which are not from leaders of i,
        for jj in 1:N
            if jj in leaders
                continue
            end
            ws[i] = vcat(ws[i], zs[jj])
        end

        #                        (2) λs of i and its followers, and
        for jj in BFSIterator(G, i)
            ws[i] = vcat(ws[i], λs[jj])
        end

        #                        (3) μs associated with i's follower policies.
        for jj in leaders
            ws[i] = vcat(ws[i], μs[(jj, i)])
        end
        println("ws for P$i: ", ws[i])
    end

    Ls

    # Construct

end