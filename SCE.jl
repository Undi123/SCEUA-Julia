# SCE
using UnPack
using Plots
include("CCE.jl")

mutable struct Pop
    position::Vector{Float64}
    cost::Float64
end

# CCE parameters
mutable struct CCE_params
    q::Int64
    alpha::Int64
    beta::Int64
    lb::Vector{Float64}
    ub::Vector{Float64}
    obj_func
end

# SCE parameters
mutable struct SCE_params
    max_iter::Int64
    n_complex::Int64
    n_complex_pop::Int64
    dim::Int64
    lb::Vector{Float64}
    ub::Vector{Float64}
    obj_func
end

import Base.isless
isless(a::Pop, b::Pop) = isless(a.cost, b.cost)

function cost_func(x)
    return sum(x.^2)
end

function uniform_rand(lb::Array{Float64, 1}, ub::Array{Float64, 1})
    dim = length(lb)
    arr = rand(dim) .* (ub .- lb) .+ lb
    return arr
end
# SCE parameters
max_iter = 500
n_complex = 5
n_complex_pop = 10
dim = 10
lb = ones(dim) * -10
ub = ones(dim) * 10
obj_func = cost_func
n_complex_pop = max(n_complex_pop, dim+1) # 如果n_pop_complex小于nVar+1,将n_pop_complex设为nVar+1
# 这个设定来自Nelder-Mead Standard
sce_params = SCE_params(max_iter, n_complex, n_complex_pop, dim, lb, ub, obj_func)

# CCE parameters
cce_q = max(round(Int32, 0.5*n_complex_pop), 2)
cce_alpha = 3
cce_beta = 5

cce_params = CCE_params(cce_q, cce_alpha, cce_beta, lb, ub, obj_func)

function SCE(sce_params, cce_params)
    @unpack max_iter, n_complex, n_complex_pop, dim, lb, ub, obj_func = sce_params

    n_pop = n_complex * n_complex_pop
    I = reshape(1:n_pop, n_complex, :)

    # Step 1. Generate rand_sample
    best_costs = Vector{Float64}(undef, max_iter)

    pops = []
    for i in 1:n_pop
        pop_position = uniform_rand(lb, ub)
        pop_cost = obj_func(pop_position)
        pop = Pop(pop_position, pop_cost)
        push!(pops, pop)
    end
    complex = Array{Pop}(undef, n_complex_pop, n_complex)

    # Step 2. Rank Points
    sort!(pops)
    best_pop = pops[1]

    # Main loop
    for iter in 1:max_iter

        # Step 3. Partion into complexes
        for j in 1:n_complex
            complex[:,j] = deepcopy(pops[I[j,:]])
            # Step 4. Evolve complex, run CCE
            complex[:,j] = CCE(complex[:,j], cce_params)
            pops[I[j,:]] = deepcopy(complex[:,j])
        end
        # Step 5. Shuffle Complexes

        sort!(pops)
        best_pop = pops[1]

        best_costs[iter] = best_pop.cost

        # Show Iteration Information
        println("Iter = ", iter)
        println("The Best Cost is: ", best_costs[iter])
    end
    best_costs
end

best_costs = SCE(sce_params, cce_params)

plot(best_costs, yaxis=:log, label = "cost")

savefig("Julia.png")
