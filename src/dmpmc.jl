using DrWatson
using Optim
using Distributions
using LinearAlgebra
using ForwardDiff

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

function rapid_mvn_prec(x::Vector{Float64}, mu::Vector{Float64}, i_sigma::Matrix{Float64}, isq_d_sigma::Float64)
    k = size(mu, 1)

    t1 = (2π)^(-k/2)
    t2 = isq_d_sigma
    lt3 = transpose(x - mu) * i_sigma * (x - mu)
    t3 = exp(-0.5 * lt3)
    return (t1 * t2 * t3)
end

function dm_pmc_step(
    target::Function,
    proposal_functions::Array,
    locations::Array,
    scales::Array,
    samples_each::Int64,
    target_dimensions::Int64 = 0,
)
    #N in literature
    num_proposals::Int64 = size(proposal_functions, 1)

    #N*K in literature
    total_samples = samples_each * num_proposals

    proposal_samples = zeros(target_dimensions, total_samples)

    i_sigma = Array{Array,1}(undef, num_proposals)
    isq_d_sigma = Array{Float64,1}(undef, num_proposals)

    weights = zeros(total_samples)
    wden = 0.

    for n = 1:num_proposals
        i_sigma[n] = inv(scales[n])
        isq_d_sigma[n] = 1/sqrt(det(scales[n]))
    end

    for n = 1:num_proposals
        proposal = proposal_functions[n]
        location = locations[n]
        scale = scales[n]
        offset = (n-1) * samples_each
        for k = 1:samples_each
            xnj = proposal_samples[:, k+offset] = rand(proposal(location, scale))
            weights[k+offset] = target(xnj)
            wden = 0.
            for d = 1:num_proposals
                wden += pdf(proposal_functions[d](locations[d], scales[d]), xnj)
            end
            wden /= num_proposals
            weights[k+offset] /= wden
        end
    end

    weights ./= sum(weights)

    new_locations_ind = wsample(1:total_samples, weights, num_proposals; replace = true)
    new_locations = Array{Array,1}(undef, num_proposals)
    for n = 1:num_proposals
        new_locations[n] = proposal_samples[:, new_locations_ind[n]]
    end

    return (proposal_samples, weights, new_locations, scales)
end

function dm_pmc_step_hadapt(
    target::Function,
    proposal_functions::Array,
    locations::Array,
    scales::Array,
    samples_each::Int64,
    target_dimensions::Int64 = 0,
)
    #N in literature
    num_proposals::Int64 = size(proposal_functions, 1)

    #N*K in literature
    total_samples = samples_each * num_proposals

    proposal_samples = zeros(target_dimensions, total_samples)

    i_sigma = Array{Array,1}(undef, num_proposals)
    isq_d_sigma = Array{Float64,1}(undef, num_proposals)

    weights = zeros(total_samples)
    wden = 0.

    # for n = 1:num_proposals
        # i_sigma[n] = inv(scales[n])
        # isq_d_sigma[n] = 1/sqrt(det(scales[n]))
    # end

    for n = 1:num_proposals
        proposal = proposal_functions[n]
        location = locations[n]
        scale = scales[n]
        offset = (n-1) * samples_each
        for k = 1:samples_each
            xnj = proposal_samples[:, k+offset] = rand(proposal(location, scale))
            weights[k+offset] = target(xnj)
            wden = 0.
            for d = 1:num_proposals
                wden += pdf(proposal_functions[d](locations[d], scales[d]), xnj)
            end
            wden /= num_proposals
            weights[k+offset] /= wden
        end
    end

    weights ./= sum(weights)

    new_locations_ind = wsample(1:total_samples, weights, num_proposals; replace = true)
    new_locations = Array{Array,1}(undef, num_proposals)
    new_scales = Array{Array,1}(undef, num_proposals)
    for n = 1:num_proposals
        new_locations[n] = proposal_samples[:, new_locations_ind[n]]
        neg_l_target(x) = -1.0 * log(target(x))
        hess = ForwardDiff.hessian(neg_l_target, locations[n])
        new_scales[n] = inv(hess)
    end


    return (proposal_samples, weights, new_locations, new_scales)
end

function dm_pmc_step_smpcov(
    target::Function,
    proposal_functions::Array,
    locations::Array,
    scales::Array,
    samples_each::Int64,
    target_dimensions::Int64 = 0,
)
    #N in literature
    num_proposals::Int64 = size(proposal_functions, 1)

    #N*K in literature
    total_samples = samples_each * num_proposals

    proposal_samples = zeros(target_dimensions, total_samples)

    i_sigma = Array{Array,1}(undef, num_proposals)
    isq_d_sigma = Array{Float64,1}(undef, num_proposals)

    weights = zeros(total_samples)
    wden = 0.

    # for n = 1:num_proposals
        # i_sigma[n] = inv(scales[n])
        # isq_d_sigma[n] = 1/sqrt(det(scales[n]))
    # end

    for n = 1:num_proposals
        proposal = proposal_functions[n]
        location = locations[n]
        scale = scales[n]
        offset = (n-1) * samples_each
        for k = 1:samples_each
            xnj = proposal_samples[:, k+offset] = rand(proposal(location, scale))
            weights[k+offset] = target(xnj)
            wden = 0.
            for d = 1:num_proposals
                wden += pdf(proposal_functions[d](locations[d], scales[d]), xnj)
            end
            wden /= num_proposals
            weights[k+offset] /= wden
        end
    end

    weights ./= sum(weights)

    new_locations_ind = wsample(1:total_samples, weights, num_proposals; replace = true)
    new_locations = Array{Array,1}(undef, num_proposals)
    new_scales = Array{Array,1}(undef, num_proposals)

    wsx = wsample(1:total_samples, weights, total_samples; replace = true)
    wsx = proposal_samples[:, wsx]
    for n = 1:num_proposals
        new_locations[n] = proposal_samples[:, new_locations_ind[n]]
        new_scales[n] = 1.3 .* cov(wsx, dims = 2) #sneaky variance inflation
    end


    return (proposal_samples, weights, new_locations, new_scales)
end
