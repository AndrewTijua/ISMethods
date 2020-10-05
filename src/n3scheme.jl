using DrWatson
using LinearAlgebra

function n3sample(target::Function, proposal_functions_full::Array, samples_each::Array, target_dimension::Int64 = 0)
    if target_dimensions <= 0
        target_dimensions = size(target(0.0), 1)
    end

    num_proposals = size(proposal_functions_full, 1)
    total_samples = samples_each * num_proposals

    proposal_samples = zeros(target_dimensions, total_samples)

    weights = zeros(total_samples)

    sampling_index = repeat(1:num_proposals, samples_each)

    for i = 1:total_samples
        propind = sampling_index[i]
        xi = proposal_samples[:, i] = rand(proposal_functions_full[propind])
        num::Float64 = 0.
        denom::Float64 = 0.
        num = target(xi)
        for d = 1:num_proposals
            denom += (1. / num_proposals) * pdf(proposal_functions[d], xi)
        end
        weights[i] = num / denom
    end

    return (proposal_samples, weights)
end

function stratified_resample(x::Matrix{Float64}, weights::Vector{Float64})
    n = size(weights, 1)
    positions = (rand(n) + collect(0:(n-1))) / n
    indexes = zeros(Int64, n)
    cs = cumsum(weights)
    (i, j) = (1, 1)
    while i <= n
        if positions[i] < cs[j]
            indexes[i] = j
            i += 1
        else
            j += 1
        end
    end
    return x[:, indexes]
end
