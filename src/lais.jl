using DrWatson
using AdvancedHMC
using Distributions
using LinearAlgebra
using ForwardDiff


# this is just going to end up worse than HMC sampling, probably best to use MH for simplicity, maybe implement single step hmc at some other point
function RWIS_draw_means(
    target::Function,
    target_dimensions::Int64,
    times::Int64,
    number_proposals::Int64,
    init_means::Array;
    n_adapts::Int64 = 1_000,
)
    nchains = number_proposals
    chains = Vector{Any}(undef, nchains)

    Threads.@threads for i = 1:nchains
        initial_θ = init_means[i]
        lp(θ) = log(target(θ))

        metric = DiagEuclideanMetric(target_dimensions)
        hamiltonian = Hamiltonian(metric, lp, ForwardDiff)

        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator = Leapfrog(initial_ϵ)

        proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
        # proposal = StaticTrajectory(integrator, n_adapts + times)
        # proposal = NUTS{SliceTS,GeneralisedNoUTurn}(integrator)
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

        samples, stats = sample(hamiltonian, proposal, initial_θ, times, adaptor, n_adapts; progress = true)
        chains[i] = samples
    end

    return chains
end

function RWIS_step(
    target::Function,
    proposal_functions::Array,
    samples_each::Int64,
    target_dimensions::Int64,
    proposal_fixed_params::Array,
    step_mean::Array,
)
    num_proposals = size(proposal_functions, 1)

    total_samples = samples_each * num_proposals

    proposal_samples = zeros(target_dimensions, total_samples)

    sampling_index = repeat(1:num_proposals, samples_each)
    weights = zeros(total_samples)

    for i = 1:total_samples
        propind = sampling_index[i]
        xi = proposal_samples[:, i] = rand(proposal_functions[propind](step_mean[propind], proposal_fixed_params[propind]))
        num::Float64 = 0.0
        denom::Float64 = 0.0
        num = target(xi)
        for d = 1:num_proposals
            denom += (1.0 / num_proposals) * pdf(proposal_functions[d](step_mean[propind], proposal_fixed_params[d]), xi)
        end
        weights[i] = num / denom
    end
    n_weights = weights ./ sum(weights)

    return (proposal_samples, n_weights)
end
