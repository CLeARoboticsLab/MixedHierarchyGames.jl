
function get_lq_kkt_conditions(G::SimpleDiGraph,
	Js::Dict{Int, Any},
	zs,
	λs,
	μs::Dict{Tuple{Int, Int}, Any},
	gs,
	ws::Dict{Int, Any},
	ys::Dict{Int, Any},
	θ;
	verbose = false,
	to = TimerOutput())
	"""
	Constructs the KKT conditions for each player in an N-player LQ Stackelberg hierarchy, considering the
	information structure provided.

	Parameters
	----------
	G (SimpleDiGraph) : The information structure of the game, defined as a directed graph.
	Js (Dict{Int, Any}) : A dictionary mapping player indices to their objective functions.
	zs (Vector{Vector{Num}}) : A vector of each player's decision variable symbols.
	λs (Vector{Vector{Num}}) : A vector of each player's Lagrange multiplier symbols for their constraints.
	μs (Dict{Tuple{Int, Int}, Vector{Num}}) : A dictionary of Lagrange multiplier symbols for each leader-follower pair.
	gs (Vector{Function}) : A vector of equality constraint functions for each player.
	ws (Dict{Int, Vector{Num}}) : A dictionary of each player's remaining variable symbols (decision output).
	ys (Dict{Int, Vector{Num}}) : A dictionary of each player's information variable symbols (decision input).
	θs (Dict{Int, Vector{Num}}) : The parameters symbols.
	verbose (Bool) : Whether to print verbose output (default: false).
	to (TimerOutput) : A TimerOutput object for performance profiling (default: TimerOutput()).

	Returns
	-------
	πs (Dict{Int, Any}) : A dictionary containing the KKT conditions for each player.
	Ms (Dict{Int, Any}) : A dictionary of M matrices (from M w + N y = 0) for each agent with a leader.
	Ns (Dict{Int, Any}) : A dictionary of N matrices (from M w + N y = 0) for each agent with a leader.
	K_evals (nothing) : Placeholder for future K evaluation results; always nothing for the LQ case.
	"""

	# Values computed by this function.
	Ms = Dict{Int, Any}()
	Ns = Dict{Int, Any}()
	Ks = Dict{Int, Any}()
	πs = Dict{Int, Any}()
	Φs = Dict{Int, Any}()

	# Compute reverse topological order to construct lagrangians and KKT conditions from leaves to root.
	order = reverse(topological_sort(G))

	if verbose
		println("Topological order of vertices:")
		for ii in order
			println(ii)
		end
	end

	for ii in order
		zi_size = length(zs[ii])

		# Include the objective of the player and its constraint term.
		Lᵢ = Js[ii](zs..., θ) - λs[ii]' * gs[ii](zs[ii])

		# If the current player Pii has no followers, then the KKT conditions consist only of
		# 1. ∇zᵢLᵢ  = 0: Stationarity of its own Lagrangian w.r.t its own variables, and
		# 2. gᵢ(zᵢ) = 0: Its own constraints.
		if is_leaf(G, ii)
			@timeit to "[KKT Conditions] Leaf" begin
				πs[ii] = vcat(Symbolics.gradient(Lᵢ, zs[ii]), # stationarity of follower only w.r.t its own vars
							  gs[ii](zs[ii]))				  # constraints for current player
			end

		# If Pii has followers, then add the follower's constraint terms to the Lagrangian, which
		# requires looking up/computing/extracting ∇wⱼΦʲ(wⱼ) for all followers j.
		else
			@timeit to "[KKT Conditions] Non-Leaf" begin
				# For players with followers, we need to add the policy constraint terms of each follower j to the Lagrangian.
				# Iterate in breadth-first order over the followers so that we can finish up the computation.
				for jj in BFSIterator(G, ii)

					# Skip the current player.
					if ii == jj
						continue
					end

					# Compute the policy term of follower j (TODO: Add a look up for efficiency).
					πⱼ = πs[jj] # This term always exists if we are proceeding in reverse topological order.

					# If the policy exists for follower j, then look up its ∇wⱼΦʲ(wⱼ) and 
					# extract ∇zᵢΦʲ(wⱼ) from it (i is a leader of j).
					# If it doesn't exist, then compute it using Mⱼ and Nⱼ and extract z̃ⱼ = [0... I ...0] ∇zᵢΦʲ(wⱼ) w₍.

					# TODO: Implement automatic extract computation using sizes of ws and ys.
					# zᵢ is often the first element of wⱼ, so we can just extract the relevant rows.
					# BUG: This assumes that zᵢ is the first element of wⱼ, which is not always true (Nash KKT combinations).

					extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))

					# SUPPOSE: we have these values which list the symbols and their sizes.
					# ws_ordering;
					# ws_sizes;

					# If we are not provided a current estimate of z, then evaluate symbolically.
					@timeit to "[KKT Conditions][Non-Leaf][Symbolic M '\' N]" begin
						# Solve Mw + Ny = 0 using symbolic M and N.
						Φʲ = - extractor * Ks[jj] * ys[jj]
					end

					Lᵢ -= μs[(ii, jj)]' * (zs[jj] - Φʲ)
				end
			end
		end

		# Once we have the Lagrangian constructed, we compute the KKT conditions by traversing in breadth-first order.
		πᵢ = []
		for jj in BFSIterator(G, ii)
			@timeit to "[KKT Conditions] Compute πᵢ" begin
				# Note: the first jj should be ii itself, followed by each follower.
				πᵢ = vcat(πᵢ, Symbolics.gradient(Lᵢ, zs[jj]))

			# Add the policy constraint.
			if ii != jj
				# println(ii, " ",jj)
				# TODO: Do we need the constant term if we add this constraint in?
				extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
				Φʲ = - extractor * Ks[jj] * ys[jj]
				πᵢ = vcat(πᵢ, zs[jj] - Φʲ)
				# println("player $ii adding policy constraint for follower $jj: ", zs[jj] - Φʲ)
				end
			end
		end
		πᵢ = vcat(πᵢ, gs[ii](zs[ii])) # Add the player's own constraints at the end.
		πs[ii] = πᵢ

		# Compute necessary terms for ∇ᵢπᵢ = Mᵢ wᵢ + Nᵢ yᵢ = 0 (if there is a leader), else not needed.
		if has_leader(G, ii)
			@timeit to "[KKT Conditions] Compute M and N for follower $ii" begin
				Ms[ii] = Symbolics.jacobian(πs[ii], ws[ii])
				Ns[ii] = Symbolics.jacobian(πs[ii], ys[ii])

				Ks[ii] = Ms[ii] \ Ns[ii] # Policy matrix for follower ii.
			end
		end
	end

	for ii in 1:nv(G)
		println("Number KKT conditions constructed for player $ii: $(length(πs[ii])).")
	end
	return πs, Ms, Ns, (;K_evals = nothing)
end


function strip_policy_constraints(πs, G, zs, gs)
	"""
	Return a copy of πs with follower policy-constraint rows removed.
	Assumes πᵢ is ordered as [∇_{zᵢ}Lᵢ; ∇_{zⱼ}Lᵢ; ...; (policy constraints); gᵢ(zᵢ)].
	"""
	πs_stripped = Dict{Int, Any}()
	for ii in 1:nv(G)
		πᵢ = πs[ii]
		parts = Any[]
		idx = 1
		for jj in BFSIterator(G, ii)
			len_z = length(zs[jj])
			push!(parts, πᵢ[idx:(idx + len_z - 1)])
			idx += len_z
			if ii != jj
				# Skip the policy-constraint block for follower jj.
				idx += len_z
			end
		end
		len_g = length(gs[ii](zs[ii]))
		push!(parts, πᵢ[idx:(idx + len_g - 1)])
		idx += len_g
		@assert idx - 1 == length(πᵢ) "strip_policy_constraints: unexpected π length for player $ii."
		πs_stripped[ii] = vcat(parts...)
	end
	return πs_stripped
end


function construct_augmented_variables(ii, all_variables, K_syms, G)
	"""
	Constructs an augmented list of variables including symbolic M and N matrices for use in optimized KKT solving.
	This vector can not have extra terms because of the dependencies among the evaluated M and N matrices.
	"""
	# If ii is a leaf, then any computations can be completed with the main set of variables.
	if is_leaf(G, ii)
		return all_variables
	end

	# If ii is not a leaf, we need to include the symbolic M and N matrices.
	augmented_variables = copy(all_variables)
	
	for jj in BFSIterator(G, ii)
		if has_leader(G, jj)
			# Vectorize them for storage.
			vcat(augmented_variables, reshape(K_syms[jj], :))
		end
	end

	return augmented_variables
end


function setup_approximate_kkt_solver(G, Js, zs, λs, μs, gs, ws, ys, θs, all_variables, backend;
	to = TimerOutput(),
	verbose = false)
	"""
	Precomputes symbolic KKT conditions, and functions for evaluating M and N matrices for each player in the game.

	Intended for use in a nonlinear solver that can take advantage of precomputed M and N matrices evaluated at z_est.
	This sets up an augmented symbolic system where each Mi and Ni depend only on symbolic versions of the M and N 
	matrices of its followers.

	Parameters
	----------
	G (SimpleDiGraph) : The information structure of the game, defined as a directed graph
	Js (Dict{Int, Any}) : A dictionary mapping player indices to their objective functions.
	zs (Vector{Vector{Num}}) : A vector of each player's decision variable symbols.
	λs (Vector{Vector{Num}}) : A vector of each player's Lagrange multiplier symbols for their constraints.
	μs (Dict{Tuple{Int, Int}, Vector{Num}}) : A dictionary of Lagrange multiplier symbols for each leader-follower pair.
	gs (Vector{Function}) : A function returning vector of equality constraint functions for each player.
	ws (Dict{Int, Vector{Num}}) : A dictionary of each player's remaining variable symbols (decision output).
	ys (Dict{Int, Vector{Num}}) : A dictionary of each player's information variable symbols (decision input).
	θs (Dict{Int, Vector{Num}}) : The parameters symbols.
	all_variables (Vector{Num}}) : A vector of all symbolic variables in the game.
	backend : The symbolic backend to use for constructing symbolic variables and functions.
	to (TimerOutput) : A TimerOutput object for performance profiling (default: TimerOutput()).
	verbose (Bool) : Whether to print verbose output (default: false).

	Returns
	-------
	out_all_augmented_variables (Vector{Num}) : A vector of all symbolic variables in the game, including symbolic K matrices.
	out (NamedTuple) : A named tuple containing:
		graph (SimpleDiGraph) : The information structure of the game, included for convenience.
		πs (Dict{Int, Any}) : A dictionary containing the KKT conditions for each player.
		K_syms (Dict{Int, Any}) : A dictionary of symbolic K matrices for each agent with a leader.
		M_fns (Dict{Int, Any}) : A dictionary of functions for evaluating M matrices for each agent with a leader.
		N_fns (Dict{Int, Any}) : A dictionary of functions for evaluating N matrices for each agent with a leader.
		π_sizes (Dict{Int, Any}) : A dictionary of sizes of KKT conditions for each player.
	"""
	N = nv(G) # number of players
	H = 1 # open-loop for now

	reverse_topological_order = reverse(topological_sort(G))

	π_sizes = Dict{Int, Any}()

	K_syms = Dict{Int, Any}()
	πs = Dict{Int, Any}()

	M_fns = Dict{Int, Any}()
	N_fns = Dict{Int, Any}()

	augmented_variables = Dict{Int, Any}()

	for ii in reverse_topological_order
		# TODO: This whole optimization is for computing M and N, which is only for players with leaders.
		#       Look into skipping players without leaders. 

		# TODO: Can be made more efficient if needed.
		# πⁱ has size num constraints + num primal variables of i AND its followers.
		π_sizes[ii] = length(gs[ii](zs[ii]))
		for jj in BFSIterator(G, ii) # loop includes ii itself.
			π_sizes[ii] += length(zs[jj])
		end

		if has_leader(G, ii)
			# TODO: Use this directly instead of Msym and Nsym, for optimization.
			K_syms[ii] = reshape(SymbolicTracingUtils.make_variables(
				backend,
				make_symbolic_variable(:K, ii, H),
				length(ws[ii]) * length(ys[ii]),
			), length(ws[ii]), length(ys[ii]))
		else
			K_syms[ii] = Symbolics.Num[]
		end

		# Build the Lagrangian using these variables.
		Lᵢ = Js[ii](zs..., θs[ii]) - λs[ii]' * gs[ii](zs[ii])
		for jj in BFSIterator(G, ii)
			if ii == jj
				continue
			end

			# TO avoid nonlinear equation solving, we encode the policy constraint using the symbolic K expression.
			zi_size = length(zs[ii])
			extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
			Φʲ = - extractor * K_syms[jj] * ys[jj]
			Lᵢ -= μs[(ii, jj)]' * (zs[jj] - Φʲ)
		end

		# Compute the KKT conditions [∇ᵢπᵢ; ⋯; ∇ⱼπᵢ; ⋯; gᵢ(zᵢ)] based on the Lagrangians.
		πᵢ = []
		for jj in BFSIterator(G, ii)
			@timeit to "[KKT Precompute] Compute πᵢ" begin
				# Note: the first jj should be ii itself, followed by each follower.
				πᵢ = vcat(πᵢ, Symbolics.gradient(Lᵢ, zs[jj]))

				# Add the policy constraint.
				if ii != jj
					# TODO: Do we need the constant term if we add this constraint in?
					zi_size = length(zs[ii])
					extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
					Φʲ = - extractor * K_syms[jj] * ys[jj]
					πᵢ = vcat(πᵢ, zs[jj] - Φʲ)
				end
			end
		end
		πᵢ = vcat(πᵢ, gs[ii](zs[ii]))
		πs[ii] = πᵢ


		# Finally, we compute symbolic versions of M and N that only depend on the symbolic versions of lower-level algorithms.
		# This allows us to evaluate M and N at any z_est without needing to recompute the entire symbolic gradient.
		# In general, solving M \ N requires a nonlinear solve. To keep things linear, we do computations using a symbolic variable K.
		# We then evaluate M and N at a given z_est, and compute K = M \ N numerically before substituting the evaluation into K for later 
		# computations.
		if has_leader(G, ii)
			@timeit to "[KKT Precompute] Compute M and N for player $ii" begin
				Mᵢ = Symbolics.jacobian(πᵢ, ws[ii])
				Nᵢ = Symbolics.jacobian(πᵢ, ys[ii])
				# TODO: Explore adding a solve for K here.
			end

			@timeit to "[KKT Precompute] Compute M, N functions for player $ii" begin
				augmented_variables[ii] = construct_augmented_variables(ii, all_variables, K_syms, G)
				M_fns[ii] = SymbolicTracingUtils.build_function(Mᵢ, augmented_variables[ii]; in_place = false)
				N_fns[ii] = SymbolicTracingUtils.build_function(Nᵢ, augmented_variables[ii]; in_place = false)
			end
		else
			# TODO: dirty code, clean up
			augmented_variables[ii] = all_variables
		end
	end

	# Identify all augmented variables.
	out_all_augmented_variables = vcat(all_variables, vcat(map(ii -> reshape(K_syms[ii], :), 1:N))...)

	return out_all_augmented_variables, (; graph=G, πs=πs, K_syms=K_syms, M_fns=M_fns, N_fns=N_fns, π_sizes=π_sizes)
end
