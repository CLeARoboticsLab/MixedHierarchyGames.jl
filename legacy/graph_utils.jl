using Graphs

#### GRAPH UTILITIES ####

function has_leader(graph::SimpleDiGraph, node::Int)
	return !is_root(graph, node)
end

function is_root(graph::SimpleDiGraph, node::Int)
	return iszero(indegree(graph, node))

end

function get_roots(graph::SimpleDiGraph)
	# Find all vertices with no incoming edges (roots).
	return [v for v in vertices(graph) if is_root(graph, v)]
end

# TODO: Replace this call with a built-in search ordering.
function get_all_leaders(graph::SimpleDiGraph, node::Int)
	parents_path = []

	parents = inneighbors(graph, node)

	while !isempty(parents)
		# Identify and save each parent until we reach the root.
		# TODO: We assume this is a tree for now. To generalize, we need to handle multiple parents 
		#       and check if the parent is already in the tree.

		parent = only(parents)
		push!(parents_path, parent)

		# Get next grandparent.
		parents = inneighbors(graph, parent)
	end

	return reverse(parents_path)
end

# TODO: Replace this call with a built-in search ordering.
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

function is_leaf(graph::SimpleDiGraph, node::Int)
	return outdegree(graph, node) == 0
end


