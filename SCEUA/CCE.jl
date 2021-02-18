using StatsBase
using UnPack

function rand_sample(P, q)
    L = Vector{Int64}(undef, q)
    for i in 1:q
        L[i] = sample(1:length(P), Weights(P))
        # L[i] = sample(1:sizeof(P), weights(P), 1, replace=true)
    end
    return L
end

function not_in_search_space(position, lb, ub)
    return any(position .<= lb) || any(position .>= ub)
end

function CCE(complex_pops, cce_params)
    # Step 1. Initialize
    @unpack q, alpha, beta, lb, ub, obj_func = cce_params
    n_pop = length(complex_pops)

    # Step 2. Assign weights
    P = [2*(n_pop+1-i) / (n_pop*(n_pop+1)) for i in 1:n_pop]

    # Calculate Population Range (Smallest Hypercube)
    new_lb = complex_pops[1].position
    new_ub = complex_pops[1].position
    for i in 2:n_pop
        new_lb = min.(new_lb, complex_pops[i].position)
        new_ub = max.(new_ub, complex_pops[i].position)
    end

    # CCE main loop
    for it in 1:beta
        # Step 3. Select parents
        L = rand_sample(P, q)
        B = complex_pops[L]

        # Step 4. Generate Offspring
        for k in 1:alpha
            # a) Sort population
            sorted_indexs = sortperm(B)
            sort!(B)
            L[:] = L[sorted_indexs]

            # Calculate the centroid
            g = zeros(length(lb))
            for i in 1:q-1
                g .= g .+ B[i].position
            end
            g .= g ./ (q-1)

            # b) Reflection step
            reflection = deepcopy(B[end])
            reflection.position = 2 .* g .- B[end].position # newly generated point using reflection
            if not_in_search_space(reflection.position, lb, ub)
                reflection.position = uniform_rand(new_lb, new_ub)
            end
            reflection.cost = obj_func(reflection.position)

            if reflection.cost < B[end].cost
                B[end] = deepcopy(reflection)
            else # Contraction
                contraction = deepcopy(B[end])
                contraction.position = (g .+ B[end].position) ./ 2
                contraction.cost = obj_func(contraction.position)

                if contraction.cost < B[end].cost
                    B[end] = deepcopy(contraction)
                else
                    B[end].position = uniform_rand(new_lb, new_ub)
                    B[end].cost = obj_func(B[end].position)
                end
            end

        end

        complex_pops[L] = B
    end
    return complex_pops
end
