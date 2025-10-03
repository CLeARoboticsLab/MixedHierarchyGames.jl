
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
	all_variables = nothing,
	verbose = false,
	to = TimerOutput())

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

	# Construct a list of all variables in order and solve.
	# TODO: Make a helper fn for getting all variables.
	if isnothing(all_variables)
		@timeit to "[KKT Conditions] Construct All Variables" begin
			temp = vcat(collect(values(μs))...)
			all_variables = vcat(vcat(zs...), vcat(λs...))
			if !isempty(temp)
				all_variables = vcat(all_variables, vcat(collect(values(μs))...))
			end
		end
	end

	M_fns = Dict{Int, Any}()
	N_fns = Dict{Int, Any}()
	for ii in order
		if has_leader(G, ii)
			M_fns[ii] = nothing
			N_fns[ii] = nothing
		end
	end

	# Values that don't need recomputing.

	# Ms_eval = Dict{Int, Any}()
	# Ns_eval = Dict{Int, Any}()

	# # Construct properly sized zeroed M and N as flattened vectors for each player for in place evaluation.
	# # This requires identifying the sizes of each M and N matrix first.
	# for ii in order
	# 	Ms_eval[ii] = zeros(length(πs[ii])*length(ws[ii]))
	# 	Ns_eval[ii] = zeros(length(πs[ii])*length(ys[ii]))
	# end

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
						@timeit to "[KKT Conditions][Non-Leaf][Symbolic M '\' N]" begin
							# Solve Mw + Ny = 0 using symbolic M and N.
							Φʲ = - extractor * (Ms[jj] \ Ns[jj]) * ys[jj] # TODO: Fill in the blank with a lookup?
						end
					else
						# TODO: Run with inplace true to do it more quickly?

						# Ensure the length of z_est matches the number of all variables.
						@assert length(z_est) == length(all_variables) "Length of z_est does not match number of all variables."

						@timeit to "[KKT Conditions][Non-Leaf][Numeric][Evaluate M]" begin
							
							# Build function for substituting values for symbolic M and N matrices.
							if isnothing(M_fns[jj])
								# Main.@infiltrate
								M_fns[jj] = SymbolicTracingUtils.build_function(Ms[jj], all_variables; in_place = false)
							end
							M_fn = M_fns[jj]

							if isnothing(N_fns[jj])
								N_fns[jj] = SymbolicTracingUtils.build_function(Ns[jj], all_variables; in_place = false)
							end
							N_fn = N_fns[jj]
						
							# # TODO: Get current estimate of z (probably all variables not just primals) as input.
							# z_estimate = z_est #zeros(length(all_variables))

							# Evaluate M at current estimate, reshaping to counteract automatic flattening.
							M_jj_eval = reshape(M_fn(z_est), size(Ms[jj]))
						end

						@timeit to "[KKT Conditions][Non-Leaf][Numeric][Evaluate N]" begin
							# Build function for substituting values for symbolic M and N matrices.
							N_fn = SymbolicTracingUtils.build_function(Ns[jj], all_variables; in_place = false)

							# Evaluate N at current estimate, reshaping to counteract automatic flattening.
							N_jj_eval = reshape(N_fn(z_est), size(Ns[jj]))
						end

						@timeit to "[KKT Conditions][Non-Leaf][Numeric][Solve M '\' N]" begin
							# Solve Mw + Ny = 0 using approximated M and N values at z_estimate.
							Φʲ = - extractor * (M_jj_eval \ N_jj_eval) * ys[jj]
						end
					end

					# TODO: Cache the result for later leaders.
					# Φs[jj] = Φʲ
					# end

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

				# if isnothing(M_fns[ii])
				# 	M_fns[ii] = SymbolicTracingUtils.build_function(Ms[ii], all_variables; in_place = false)
				# end
				# if isnothing(N_fns[ii])
				# 	N_fns[ii] = SymbolicTracingUtils.build_function(Ns[ii], all_variables; in_place = false)
				# end

				# TODO: For more efficient caching, do the M \ N computation here and store the result.
			end
		end
	end

	return πs, Ms, Ns, (;K_evals = nothing)
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

# function setup_augmented_kkt_system(G, Js, zs, λs, μs, gs, ws, ys, θ, all_variables, backend; to=TimerOutput(), verbose = false)
# 	"""
# 	Intended for use in a nonlinear solver that can take advantage of precomputed M and N matrices evaluated at z_est.
# 	This sets up an augmented symbolic system where each Mi and Ni depend only on symbolic versions of the Kj matrices
# 	of its followers.
# 	"""
# 	N = nv(G) # number of players
# 	H = 1 # open-loop for now

# 	reverse_topological_order = reverse(topological_sort(G))

# 	πs = Dict{Int, Any}()
# 	π_sizes = Dict{Int, Any}()

# 	K_syms = Dict{Int, Any}()

# 	M_fns = Dict{Int, Any}()
# 	N_fns = Dict{Int, Any}()

# 	augmented_variables = Dict{Int, Any}()

# 	for ii in reverse_topological_order
# 		# TODO: This whole optimization is for computing M and N, which is only for players with leaders.
# 		#       Look into skipping players without leaders. 

# 		# TODO: Can be made more efficient if needed.
# 		# πⁱ has size num constraints + num primal variables of i AND its followers.
# 		π_sizes[ii] = length(gs[ii](zs[ii]))
# 		for jj in BFSIterator(G, ii) # loop includes ii itself.
# 			π_sizes[ii] += length(zs[jj])
# 		end

# 		if has_leader(G, ii)
# 			# TODO: Use this directly instead of Msym and Nsym, for optimization.
# 			K_syms[ii] = reshape(SymbolicTracingUtils.make_variables(
# 				backend,
# 				make_symbolic_variable(:K, ii, H),
# 				length(ws[ii]) * length(ys[ii]),
# 			), length(ws[ii]), length(ys[ii]))
# 		else
# 			K_syms[ii] = nothing
# 		end

# 		# Build the Lagrangian using these variables.
# 		Lᵢ = Js[ii](zs..., θ) - λs[ii]' * gs[ii](zs[ii])
# 		for jj in BFSIterator(G, ii)
# 			if ii == jj
# 				continue
# 			end

# 			# We encode the policy constraint using the symbolic K expression.
# 			# TODO: Try the symbolic K variable.
# 			# Main.@infiltrate
# 			zi_size = length(zs[ii])
# 			extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
# 			Lᵢ -= μs[(ii, jj)]' * (zs[jj] - extractor * K_syms[jj] * ys[jj])
# 		end

# 		# Compute the KKT conditions.
# 		πᵢ = []
# 		for jj in BFSIterator(G, ii)
# 			@timeit to "[KKT Precompute] Compute πᵢ" begin
# 				# Note: the first jj should be ii itself, followed by each follower.
# 				πᵢ = vcat(πᵢ, Symbolics.gradient(Lᵢ, zs[jj]))
# 			end
# 			πᵢ = vcat(πᵢ, gs[ii](zs[ii]))
# 		end
# 		πs[ii] = πᵢ

# 		# Finally, we compute symbolic versions of M and N that only depend on the symbolic versions of lower-level algorithms.
# 		# This allows us to evaluate M and N at any z_est without needing to recompute the entire symbolic gradient.
# 		if has_leader(G, ii)
# 			@timeit to "[KKT Precompute] Compute M and N for player $ii" begin
# 				Mᵢ = Symbolics.jacobian(πᵢ, ws[ii])
# 				Nᵢ = Symbolics.jacobian(πᵢ, ys[ii])
# 				# TODO: Explore adding a solve for K here.
# 			end

# 			@timeit to "[KKT Precompute] Compute M, N functions for player $ii" begin
# 				augmented_variables[ii] = construct_augmented_variables(ii, all_variables, K_syms, G)
# 				M_fns[ii] = SymbolicTracingUtils.build_function(Mᵢ, augmented_variables[ii]; in_place = false)
# 				N_fns[ii] = SymbolicTracingUtils.build_function(Nᵢ, augmented_variables[ii]; in_place = false)

# 				# TODO: Explore building K function directly.
# 			end
# 		else
# 			# TODO: dirty code, clean up
# 			augmented_variables[ii] = all_variables
# 		end
# 	end

# 	# Identify all augment variables.
# 	out_all_augment_variables = vcat(all_variables, map(ii -> reshape(@something(K_syms[ii], []), :), 1:N) |> vcat)

# 	# TODO: Write this function to compute the Ks using M_fns and N_fns in the right order, then return the values.
# 	# TODO: Need to ensure that function arguments are in the right order, or go and make every function accept all arguments with some zeroed.
# 	function compute_all_Ks(z_est)
# 		K_evals = Dict{Int, Any}()
# 		for ii in reverse_topological_order
# 			if has_leader(G, ii)
# 				# Evaluate K at current estimate, reshaping to counteract automatic flattening.
# 				K_evals[ii] = reshape(K_fns[ii](z_est), size(K_syms[ii]))
# 			end
# 		end
# 		all_vectorized_Ks = map(ii -> reshape(@something(K_evals[ii], []), :), 1:N) |> vcat
# 		return all_vectorized_Ks, K_evals
# 	end

# 	return fast_kkt_solver, out_all_augment_variables
# end


function setup_fast_kkt_solver(G, Js, zs, λs, μs, gs, ws, ys, θ, all_variables, backend; to=TimerOutput(), verbose = false)
	"""
	Intended for use in a nonlinear solver that can take advantage of precomputed M and N matrices evaluated at z_est.
	This sets up an augmented symbolic system where each Mi and Ni depend only on symbolic versions of the M and N 
	matrices of its followers.
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
		Lᵢ = Js[ii](zs..., θ) - λs[ii]' * gs[ii](zs[ii])
		for jj in BFSIterator(G, ii)
			if ii == jj
				continue
			end

			# We encode the policy constraint using the symbolic K expression.
			# TODO: Try the symbolic K variable.
			# Main.@infiltrate
			zi_size = length(zs[ii])
			extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
			Lᵢ -= μs[(ii, jj)]' * (zs[jj] - extractor * K_syms[jj] * ys[jj])
		end

		# Compute the KKT conditions.
		πᵢ = []
		for jj in BFSIterator(G, ii)
			@timeit to "[KKT Precompute] Compute πᵢ" begin
				# Note: the first jj should be ii itself, followed by each follower.
				πᵢ = vcat(πᵢ, Symbolics.gradient(Lᵢ, zs[jj]))
			end
		end
		πᵢ = vcat(πᵢ, gs[ii](zs[ii]))
		πs[ii] = πᵢ


		# Finally, we compute symbolic versions of M and N that only depend on the symbolic versions of lower-level algorithms.
		# This allows us to evaluate M and N at any z_est without needing to recompute the entire symbolic gradient.
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

				# TODO: Explore building K function directly.
			end
		else
			# TODO: dirty code, clean up
			augmented_variables[ii] = all_variables
		end
	end

	# Identify all augment variables.
	for ii in keys(K_syms)
		println("K_sym $ii: ", eltype(K_syms[ii]))
	end
	out_all_augment_variables = vcat(all_variables, vcat(map(ii -> reshape(K_syms[ii], :), 1:N))...)


	# We've completed the precomputation, now we return a function that takes a point to linearize M and N around.
	function fast_kkt_solver(z_est)
		
		# Calls the lq solver with the approximated values of M and N.
		# TODO: Rewrite to evaluate them in this block instead of passing the whole other function.
		return get_lq_kkt_conditions_new(
			G,
			Js,
			zs,
			λs,
			μs,
			gs,
			ws,
			ys,
			θ;
			linearization_point = (;
				z_est = z_est,
				M_fns = M_fns,
				N_fns = N_fns,
				augmented_variables = augmented_variables,
			),
			all_variables = all_variables,
			verbose = verbose,
			to = to,
		)		
	end

	return fast_kkt_solver, out_all_augment_variables, (; πs = πs, K_syms = K_syms, M_fns = M_fns, N_fns = N_fns, π_sizes = π_sizes)
end


function get_lq_kkt_conditions_new(
	G::SimpleDiGraph,
	Js::Dict{Int, Any},
	zs,
	λs,
	μs::Dict{Tuple{Int, Int}, Any},
	gs,
	ws::Dict{Int, Any},
	ys::Dict{Int, Any},
	θ;
	linearization_point = nothing, # should be a named tuple with z_est, M_fns, N_fns, augmented_variables, etc.
	all_variables = nothing,
	verbose = false,
	to = TimerOutput())

	# Values computed by this function.
	Ms = Dict{Int, Any}()
	Ns = Dict{Int, Any}()
	πs = Dict{Int, Any}()
	Φs = Dict{Int, Any}()

	# Compute reverse topological order to construct lagrangians and KKT conditions from leaves to root.
	reverse_topological_order = reverse(topological_sort(G))

	if verbose
		println("Topological order of vertices:")
		for ii in order
			println(ii)
		end
	end

	# Construct a list of all variables in order and solve.
	# TODO: Make a helper fn for getting all variables.
	if isnothing(all_variables)
		@timeit to "[KKT Conditions] Construct All Variables" begin
			temp = vcat(collect(values(μs))...)
			all_variables = vcat(vcat(zs...), vcat(λs...))
			if !isempty(temp)
				all_variables = vcat(all_variables, vcat(collect(values(μs))...))
			end
		end
	end

	# If specified, evaluate the M and N matrices at the provided linearization point.
	if !isnothing(linearization_point)
		z_est = linearization_point.z_est
		M_fns = linearization_point.M_fns
		N_fns = linearization_point.N_fns
		all_augmented_variables = linearization_point.augmented_variables

		# Ensure the length of z_est matches the number of all variables.
		@assert length(z_est) == length(all_variables) "Length of z_est does not match number of all variables."

		M_evals = Dict{Int, Any}()
		N_evals = Dict{Int, Any}()

		K_evals = Dict{Int, Any}()

		π_sizes = Dict{Int, Any}()

		@timeit to "[KKT Conditions][Non-Leaf][Numeric][Evaluate M]" begin

			for ii in reverse_topological_order
				# TODO: Can be made more efficient if needed.
				# πⁱ has size num constraints + num primal variables of i AND its followers.
				π_sizes[ii] = length(gs[ii](zs[ii]))
				for jj in BFSIterator(G, ii) # loop includes ii itself.
					π_sizes[ii] += length(zs[jj])
				end

				# TODO: optimize: we can use one massive augmented vector if we include dummy values for variables we don't have yet.
				# Get the list of symbols we need values for.
				augmented_variables = all_augmented_variables[ii]

				if has_leader(G, ii)
				# if is_leaf(G, ii) 
				# 	M_evals[ii] = M_fns[ii](z_est)
				# 	N_evals[ii] = N_fns[ii](z_est)
				# else
					# Create an augmented version using the numerical values that we have (based on z_est and computed follower Ms/Ns).
					augmented_z_est = map(jj -> reshape(K_evals[jj], :), collect(BFSIterator(G, ii))[2:end]) # skip ii itself
					augmented_z_est = vcat(z_est, augmented_z_est...)
					
					# Produce linearized versions of the current M and N values which can be used.
					M_evals[ii] = reshape(M_fns[ii](augmented_z_est), π_sizes[ii], length(ws[ii]))
					N_evals[ii] = reshape(N_fns[ii](augmented_z_est), π_sizes[ii], length(ys[ii]))

					K_evals[ii] = M_evals[ii] \ N_evals[ii]
				else
					M_evals[ii] = nothing
					N_evals[ii] = nothing
					K_evals[ii] = nothing
				end
			end
		end
	end
		

	# M_fns = Dict{Int, Any}()
	# N_fns = Dict{Int, Any}()
	# for ii in order
	# 	if has_leader(G, ii)
	# 		M_fns[ii] = nothing
	# 		N_fns[ii] = nothing
	# 	end
	# end

	# Values that don't need recomputing.

	# Ms_eval = Dict{Int, Any}()
	# Ns_eval = Dict{Int, Any}()

	# # Construct properly sized zeroed M and N as flattened vectors for each player for in place evaluation.
	# # This requires identifying the sizes of each M and N matrix first.
	# for ii in order
	# 	Ms_eval[ii] = zeros(length(πs[ii])*length(ws[ii]))
	# 	Ns_eval[ii] = zeros(length(πs[ii])*length(ys[ii]))
	# end

	for ii in reverse_topological_order
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

					# If we are not provided a point around which to linearize, then evaluate symbolically.
					# Note: this only works for LQ systems.
					if isnothing(linearization_point)
						@timeit to "[KKT Conditions][Non-Leaf][Symbolic M '\' N]" begin
							# Solve Mw + Ny = 0 using symbolic M and N.
							Φʲ = - extractor * (Ms[jj] \ Ns[jj]) * ys[jj] # TODO: Fill in the blank with a lookup?
						end
					else
						# TODO: Run with inplace true to do it more quickly?

					# 	@timeit to "[KKT Conditions][Non-Leaf][Numeric][Evaluate M]" begin
							
					# 		# Build function for substituting values for symbolic M and N matrices.
					# 		if isnothing(M_fns[jj])
					# 			# Main.@infiltrate
					# 			M_fns[jj] = SymbolicTracingUtils.build_function(Ms[jj], all_variables; in_place = false)
					# 		end
					# 		M_fn = M_fns[jj]

					# 		if isnothing(N_fns[jj])
					# 			N_fns[jj] = SymbolicTracingUtils.build_function(Ns[jj], all_variables; in_place = false)
					# 		end
					# 		N_fn = N_fns[jj]
						
					# 		# # TODO: Get current estimate of z (probably all variables not just primals) as input.
					# 		# z_estimate = z_est #zeros(length(all_variables))

					# 		# Evaluate M at current estimate, reshaping to counteract automatic flattening.
					# 		M_jj_eval = reshape(M_fn(z_est), size(Ms[jj]))
					# 	end

					# 	@timeit to "[KKT Conditions][Non-Leaf][Numeric][Evaluate N]" begin
					# 		# Build function for substituting values for symbolic M and N matrices.
					# 		N_fn = SymbolicTracingUtils.build_function(Ns[jj], all_variables; in_place = false)

					# 		# Evaluate N at current estimate, reshaping to counteract automatic flattening.
					# 		N_jj_eval = reshape(N_fn(z_est), size(Ns[jj]))
					# 	end

						@timeit to "[KKT Conditions][Non-Leaf][Numeric][Solve M '\' N]" begin
							# Solve Mw + Ny = 0 using approximated M and N values at z_estimate.
							Φʲ = - extractor * (M_evals[jj] \ N_evals[jj]) * ys[jj]
						end
					end

					# TODO: Cache the result for later leaders.
					# Φs[jj] = Φʲ
					# end

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

				# TODO: For slightly more efficient caching, do the M \ N computation here and store the result.
			end
		end
	end

	return πs, Ms, Ns, (;K_evals)
end


function get_lq_kkt_conditions_optimized(G::SimpleDiGraph,
	Js::Dict{Int, Any},
	zs,
	λs,
	μs::Dict{Tuple{Int, Int}, Any},
	gs,
	ws::Dict{Int, Any},
	ys::Dict{Int, Any},
	θ;
	z_est = nothing,
	all_variables = nothing,
	verbose = false,
	to = TimerOutput())

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

	# Construct a list of all variables in order and solve.
	# TODO: Make a helper fn for getting all variables.
	if isnothing(all_variables)
		@timeit to "[KKT Conditions] Construct All Variables" begin
			temp = vcat(collect(values(μs))...)
			all_variables = vcat(vcat(zs...), vcat(λs...))
			if !isempty(temp)
				all_variables = vcat(all_variables, vcat(collect(values(μs))...))
			end
		end
	end

	# Values that don't need recomputing.
	nonpolicy_Ls = Dict{Int, Any}()
	πs_cache = Dict{Int, Any}()

	# Include the objective of the player and its constraint term.
	for ii in order
		# Include the objective of the player and its constraint term. Policy can be computed later.
		Lᵢ = nonpolicy_Ls[ii] = Js[ii](zs..., θ) - λs[ii]' * gs[ii](zs[ii])

		if is_leaf(G, ii)
			# Leaves don't have policy constraints, so we can cache them ahead of time.
			@timeit to "[KKT Conditions] Leaf" begin
				πs_cache[ii] = vcat(Symbolics.gradient(Lᵢ, zs[ii]), # stationarity of follower only w.r.t its own vars
								    gs[ii](zs[ii])) 			    # constraints for current player
			end
		else
			# Non-leaves will need to recompute their Lagrangian with policy constraints, so don't cache.
			πs_cache[ii] = nothing
		end

	end




	# Ms_eval = Dict{Int, Any}()
	# Ns_eval = Dict{Int, Any}()

	# # Construct properly sized zeroed M and N as flattened vectors for each player for in place evaluation.
	# # This requires identifying the sizes of each M and N matrix first.
	# for ii in order
	# 	Ms_eval[ii] = zeros(length(πs[ii])*length(ws[ii]))
	# 	Ns_eval[ii] = zeros(length(πs[ii])*length(ys[ii]))
	# end

	for ii in order
		# Include the objective of the player and its constraint term.
		Lᵢ = nonpolicy_Ls[ii]

		# If the current player Pii has no followers, then the KKT conditions consist only of
		# 1. ∇zᵢLᵢ  = 0: Stationarity of its own Lagrangian w.r.t its own variables, and
		# 2. gᵢ(zᵢ) = 0: Its own constraints.
		if is_leaf(G, ii)
			# @timeit to "[KKT Conditions] Leaf" begin
			πs[ii] = πs_cache[ii]
			# vcat(Symbolics.gradient(Lᵢ, zs[ii]), # stationarity of follower only w.r.t its own vars
			# 	gs[ii](zs[ii])) # constraints for current player
			# end

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
						@timeit to "[KKT Conditions][Non-Leaf][Symbolic M '\' N]" begin
							# Solve Mw + Ny = 0 using symbolic M and N.
							Φʲ = - extractor * (Ms[jj] \ Ns[jj]) * ys[jj] # TODO: Fill in the blank with a lookup?
						end
					else
						@timeit to "[KKT Conditions][Non-Leaf][Numeric M '\' N]" begin
							# TODO: Use the code below to implement numeric evaluation of M and N.
							# Main.@infiltrate
							# TODO: Run with inplace true to do it more quickly?

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
					end

					# TODO: Cache the result for later leaders.
					# Φs[jj] = Φʲ
					# end

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
		πᵢ = vcat(πᵢ, gs[ii](zs[ii])) # Add player ii's (the leader's) own constraints at the end.
		πs[ii] = πᵢ

		# Compute necessary terms for ∇ᵢπᵢ = Mᵢ wᵢ + Nᵢ yᵢ = 0 (if there is a leader), else not needed.
		if has_leader(G, ii)
			@timeit to "[KKT Conditions] Compute M and N for follower $ii" begin
				Ms[ii] = Symbolics.jacobian(πs[ii], ws[ii])
				Ns[ii] = Symbolics.jacobian(πs[ii], ys[ii])

				# TODO: For more efficient caching, do the M \ N computation here and store the result.
			end
		end
	end

	return πs, Ms, Ns
end
