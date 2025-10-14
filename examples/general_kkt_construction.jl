
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
	θ (Num) : The parameter symbol.
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

					zi_size = length(zs[ii])
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
			end
		end
		πᵢ = vcat(πᵢ, gs[ii](zs[ii])) # Add the player's own constraints at the end.
		πs[ii] = πᵢ

		# Compute necessary terms for ∇ᵢπᵢ = Mᵢ wᵢ + Nᵢ yᵢ = 0 (if there is a leader), else not needed.
		if has_leader(G, ii)
			@timeit to "[KKT Conditions] Compute M and N for follower $ii" begin
				Ms[ii] = Symbolics.jacobian(πs[ii], ws[ii])
				Ns[ii] = Symbolics.jacobian(πs[ii], ys[ii])

				Ks[ii] = - Ms[ii] \ Ns[ii] # Policy matrix for follower ii.
			end
		end
	end

	return πs, Ms, Ns, (;K_evals = nothing)
end
