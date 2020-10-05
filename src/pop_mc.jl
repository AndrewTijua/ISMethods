using DrWatson
using Optim
using Distributions
using Plots
using LinearAlgebra
using InvertedIndices

function mixture_posterior_prob(X, proposal_functions::Array, proposal_weights::Vector{Float64}, parameters::Array)
    num_proposals = size(proposal_weights, 1)
    denom = zeros(num_proposals)
    for d = 1:num_proposals
        denom[d] += pdf(proposal_functions[d](parameters[d]...), X)
    end
    return denom
end

function mpmc_step(
    target::Function,
    proposal_functions::Array,
    proposal_weights::Vector{Float64},
    parameters::Array,
    samples::Int64,
    fs_reconst::Array,
    no_optim::Array,
    target_dimensions::Int64 = 0,
)
    sample_weights = zeros(samples)
    if target_dimensions <= 0
        target_dimensions = size(target(0.0), 1)
    end

    num_proposals = size(proposal_weights, 1)

    selected_proposals = wsample(1:num_proposals, proposal_weights, samples, ordered = true)

    proposal_samples = zeros(target_dimensions, samples)

    mix_post_prob = zeros(num_proposals, samples)

    for i = 1:samples
        spind = selected_proposals[i]
        proposal_samples[:, i] = rand(proposal_functions[spind](parameters[spind]...))
        xit = proposal_samples[:, i]
        mix_post_prob[:, i] = mixture_posterior_prob(xit, proposal_functions, proposal_weights, parameters)
        denom = 0.0
        for d = 1:num_proposals
            denom += pdf(proposal_functions[d](parameters[d]...), xit)
        end
        sample_weights[i] = target(xit) / denom
    end

    sample_weights ./= sum(sample_weights)

    new_proposal_weights = zeros(num_proposals)

    mpp = zeros(num_proposals)

    for i = 1:samples
        mpp = mix_post_prob[:, i]
        for d = 1:num_proposals
            new_proposal_weights[d] += (mpp[d] / sum(mpp))
        end
    end

    new_proposal_weights ./= sum(new_proposal_weights)

    function breakdown(X::Array)
        return collect(Iterators.flatten(X))
    end

    function optimise_helper(proposal_samples, parameters, sample_weights, proposal_function, mpp, rcf, nooptim)
        rv = 0.0
        pv = rcf(parameters, nooptim)
        for i = 1:size(proposal_samples, 2)
            rv -= sample_weights[i] * mpp[i] * pdf(proposal_function(pv...), proposal_samples[:, i])
        end
        return rv
    end

    new_parameters = parameters

    for d = 1:num_proposals
        pv = breakdown(parameters[d][Not(no_optim[d])])
        noopt = parameters[d][no_optim[d]]
        optimise_me(parameters) = optimise_helper(
            proposal_samples,
            parameters,
            sample_weights,
            proposal_functions[d],
            mix_post_prob[d, :],
            fs_reconst[d],
            noopt,
        )
        ores = optimize(optimise_me, pv, LBFGS())
        new_parameters[d] = fs_reconst[d](Optim.minimizer(ores), noopt)
    end
    return (proposal_samples, new_proposal_weights, new_parameters)
end
