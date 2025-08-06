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
