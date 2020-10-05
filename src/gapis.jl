using DrWatson
using Distributions
using LinearAlgebra
using ForwardDiff

function gapis_step(
    target::Function,
    proposal_functions::Array,
    locations::Array,
    scales::Array,
    samples_each::Int64,
    lambda::Float64 = 0.05,
    target_dimensions::Int64 = 0,
)

    if target_dimensions <= 0
        target_dimensions = size(target(0.0), 1)
    end

    #N in literature
    num_proposals = size(proposal_functions, 1)

    proposal_samples = zeros(target_dimensions, samples_each .* num_proposals)

    for n = 1:num_proposals
        proposal = proposal_functions[n]
        offset = (n - 1) * samples_each
        location = locations[n]
        scale = scales[n]
        for i = 1:samples_each
            proposal_samples[:, i+offset] = rand(proposal(location, scale))
        end
    end

    weights = zeros(samples_each .* num_proposals)

    for i = 1:(samples_each*num_proposals)
        c_sample = proposal_samples[:, i]
        weights[i] = target(c_sample)
        dn = 0.0
        for n = 1:num_proposals
            proposal = proposal_functions[n]
            location = locations[n]
            scale = scales[n]
            dn += (1 / num_proposals) * pdf(proposal(location, scale), c_sample)
        end
        weights[i] /= dn
    end

    new_locations = locations
    new_scales = scales

    for n = 1:num_proposals
        l_target(x) = log(target(x))
        neg_l_target(x) = -1.0 .* l_target(x)
        grad = ForwardDiff.gradient(l_target, locations[n])
        hess = ForwardDiff.hessian(neg_l_target, locations[n])
        new_locations[n] += lambda * grad
        new_scales[n] = inv(hess)
    end

    return (proposal_samples, new_locations, new_scales)
end
