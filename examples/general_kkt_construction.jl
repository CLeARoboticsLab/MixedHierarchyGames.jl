
function get_lq_kkt_conditions(G::SimpleDiGraph,
	Js::Dict{Int, Any},
	zs,
	λs,
	μs::Dict{Tuple{Int, Int}, Any},
	gs,
	ws::Dict{Int, Any},
	ys::Dict{Int, Any},
	θ;
	z_est = nothing,
	verbose = false)

	# Values computed by this function.
	Ms = Dict{Int, Any}()
	Ns = Dict{Int, Any}()
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
			πs[ii] = vcat(Symbolics.gradient(Lᵢ, zs[ii]), # stationarity of follower only w.r.t its own vars
				gs[ii](zs[ii])) # constraints for current player

		# If Pii has followers, then add the follower's constraint terms to the Lagrangian, which
		# requires looking up/computing/extracting ∇wⱼΦʲ(wⱼ) for all followers j.
		else

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

				# TODO: Uncomment so we have caching.
				# if jj in keys(Φs)
				#     Φʲ = Φs[jj]
				# else
				# TODO: Implement automatic extract computation using sizes of ws and ys.
				# zᵢ is often the first element of wⱼ, so we can just extract the relevant rows.
				# BUG: This assumes that zᵢ is the first element of wⱼ, which is not always true (Nash KKT combinations).

				zi_size = length(zs[ii])
				extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))

				# SUPPOSE: we have these values which list the symbols and their sizes.
				# ws_ordering;
				# ws_sizes;

				# TODO: Consider only evaluating when we don't have linear M and N. How to identify?

				# If we are not provided a current estimate of z, then evaluate symbolically.
				if isnothing(z_est)
					Φʲ = - extractor * (Ms[jj] \ Ns[jj]) * ys[jj] # TODO: Fill in the blank with a lookup?
				else
					# TODO: Use the code below to implement numeric evaluation of M and N.
					# Main.@infiltrate
					# TODO: Run with inplace true to do it more quickly?

					# Construct a list of all variables in order and solve.
					# TODO: Make a helper fn for getting all variables.
					temp = vcat(collect(values(μs))...)
					all_variables = vcat(vcat(zs...), vcat(λs...))
					if !isempty(temp)
						all_variables = vcat(all_variables, vcat(collect(values(μs))...))
					end

					# Ensure the length of z_est matches the number of all variables.
					@assert length(z_est) == length(all_variables) "Length of z_est does not match number of all variables."

					# Build function for substituting values for symbolic M and N matrices.
					M_fn = SymbolicTracingUtils.build_function(Ms[jj], all_variables; in_place = false)
					N_fn = SymbolicTracingUtils.build_function(Ns[jj], all_variables; in_place = false)

					# # TODO: Get current estimate of z (probably all variables not just primals) as input.
					# z_estimate = z_est #zeros(length(all_variables))

					# Evaluate M and N at current estimate, reshaping to counteract automatic flattening.
					M_jj_eval = reshape(M_fn(z_est), size(Ms[jj]))
					N_jj_eval = reshape(N_fn(z_est), size(Ns[jj]))

					# Solve Mw + Ny = 0 using approximated M and N values at z_estimate.
					Φʲ = - extractor * (M_jj_eval \ N_jj_eval) * ys[jj]
				end

				# TODO: Cache the result for later leaders.
				# Φs[jj] = Φʲ
				# end

				Lᵢ -= μs[(ii, jj)]' * (zs[jj] - Φʲ)
			end
		end

		# Once we have the Lagrangian constructed, we compute the KKT conditions by traversing in breadth-first order.
		πᵢ = []
		for jj in BFSIterator(G, ii)
			# Note: the first jj should be ii itself, followed by each follower.
			πᵢ = vcat(πᵢ, Symbolics.gradient(Lᵢ, zs[jj]))
		end
		πᵢ = vcat(πᵢ, gs[ii](zs[ii])) # Add the player's own constraints at the end.
		πs[ii] = πᵢ

		# Compute necessary terms for ∇ᵢπᵢ = Mᵢ wᵢ + Nᵢ yᵢ = 0.
		Ms[ii] = Symbolics.jacobian(πs[ii], ws[ii])
		Ns[ii] = Symbolics.jacobian(πs[ii], ys[ii])
	end

	return πs, Ms, Ns
end

# # TODO: Implement function that evaluates KKT conditions through numerical evaluation of M and N.
# function get_nonlinear_kkt_conditions_thru_eval(G::SimpleDiGraph,
# 	Js::Dict{Int, Any},
# 	zs,
# 	λs,
# 	μs::Dict{Tuple{Int, Int}, Any},
# 	gs,
# 	ws::Dict{Int, Any},
# 	ys::Dict{Int, Any},
# 	θ;
# 	verbose = false)

# 	# Values computed by this function.
# 	Ms = Dict{Int, Any}()
# 	Ns = Dict{Int, Any}()
# 	πs = Dict{Int, Any}()
# 	Φs = Dict{Int, Any}()

# 	# Compute reverse topological order to construct lagrangians and KKT conditions from leaves to root.
# 	order = reverse(topological_sort(G))

# 	if verbose
# 		println("Topological order of vertices:")
# 		for ii in order
# 			println(ii)
# 		end
# 	end

# 	for ii in order
# 		# Include the objective of the player and its constraint term.
# 		Lᵢ = Js[ii](zs..., θ) - λs[ii]' * gs[ii](zs[ii])

# 		# If the current player Pii has no followers, then the KKT conditions consist only of
# 		# 1. ∇zᵢLᵢ  = 0: Stationarity of its own Lagrangian w.r.t its own variables, and
# 		# 2. gᵢ(zᵢ) = 0: Its own constraints.
# 		if is_leaf(G, ii)
# 			πs[ii] = vcat(Symbolics.gradient(Lᵢ, zs[ii]), # stationarity of follower only w.r.t its own vars
# 				gs[ii](zs[ii])) # constraints for current player

# 		# If Pii has followers, then add the follower's constraint terms to the Lagrangian, which
# 		# requires looking up/computing/extracting ∇wⱼΦʲ(wⱼ) for all followers j.
# 		else

# 			# For players with followers, we need to add the policy constraint terms of each follower j to the Lagrangian.
# 			# Iterate in breadth-first order over the followers so that we can finish up the computation.
# 			for jj in BFSIterator(G, ii)

# 				# Skip the current player.
# 				if ii == jj
# 					continue
# 				end

# 				# Compute the policy term of follower j (TODO: Add a look up for efficiency).
# 				πⱼ = πs[jj] # This term always exists if we are proceeding in reverse topological order.

# 				# If the policy exists for follower j, then look up its ∇wⱼΦʲ(wⱼ) and 
# 				# extract ∇zᵢΦʲ(wⱼ) from it (i is a leader of j).
# 				# If it doesn't exist, then compute it using Mⱼ and Nⱼ and extract z̃ⱼ = [0... I ...0] ∇zᵢΦʲ(wⱼ) w₍.

# 				# TODO: Uncomment so we have caching.
# 				# if jj in keys(Φs)
# 				#     Φʲ = Φs[jj]
# 				# else
# 				# TODO: Implement automatic extract computation using sizes of ws and ys.
# 				# zᵢ is often the first element of wⱼ, so we can just extract the relevant rows.
# 				# BUG: This assumes that zᵢ is the first element of wⱼ, which is not always true (Nash KKT combinations).

# 				zi_size = length(zs[ii])
# 				extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))

# 				# SUPPOSE: we have these values which list the symbols and their sizes.
# 				# ws_ordering;
# 				# ws_sizes;


# 				# TODO: Use the code below to implement numeric evaluation of M and N.
# 				Main.@infiltrate
# 				# Φʲ = - extractor * (Ms[jj] \ Ns[jj]) * ys[jj] # TODO: Fill in the blank with a lookup?

# 				# all_πs = Vector{Symbolics.Num}(vcat(collect(values(πs))...))
# 				# # TODO: Run with inplace true to do bali bali.
# 				# π_fns = SymbolicTracingUtils.build_function(all_πs, all_variables; in_place = false)
# 				# π_eval = π_fns(z_sol)

# 				# Construct a list of all variables in order and solve.
# 				# TODO: Make a helpie fn for getting all variables.
# 				temp = vcat(collect(values(μs))...)
# 				all_variables = vcat(vcat(zs...), vcat(λs...))
# 				if !isempty(temp)
# 					all_variables = vcat(all_variables, vcat(collect(values(μs))...))
# 				end


# 				# Build function for substituting values for symbolic M and N matrices.
# 				M_fn = SymbolicTracingUtils.build_function(Ms[jj], all_variables; in_place = false)
# 				N_fn = SymbolicTracingUtils.build_function(Ns[jj], all_variables; in_place = false)

# 				# TODO: Get current estimate of z (probably all variables not just primals) as input.
# 				z_estimate = zeros(length(all_variables))

# 				# Evaluate M and N at current estimate, reshaping to counteract automatic flattening.
# 				M_jj_eval = reshape(M_fn(z_estimate), size(Ms[jj]))
# 				N_jj_eval = reshape(N_fn(z_estimate), size(Ns[jj]))

# 				# Solve Mw + Ny = 0 using approximated M and N values at z_estimate.
# 				Φʲ = - extractor * (M_jj_eval \ N_jj_eval) * ys[jj]

# 				# TODO: Cache the result for later leaders.
# 				# Φs[jj] = Φʲ
# 				# end

# 				Lᵢ -= μs[(ii, jj)]' * (zs[jj] - Φʲ)
# 			end
# 		end

# 		# Once we have the Lagrangian constructed, we compute the KKT conditions by traversing in breadth-first order.
# 		πᵢ = []
# 		for jj in BFSIterator(G, ii)
# 			# Note: the first jj should be ii itself, followed by each follower.
# 			πᵢ = vcat(πᵢ, Symbolics.gradient(Lᵢ, zs[jj]))
# 		end
# 		πᵢ = vcat(πᵢ, gs[ii](zs[ii])) # Add the player's own constraints at the end.
# 		πs[ii] = πᵢ

# 		# Compute necessary terms for ∇ᵢπᵢ = Mᵢ wᵢ + Nᵢ yᵢ = 0.
# 		Ms[ii] = Symbolics.jacobian(πs[ii], ws[ii])
# 		Ns[ii] = Symbolics.jacobian(πs[ii], ys[ii])
# 	end

# 	return πs, Ms, Ns
# end