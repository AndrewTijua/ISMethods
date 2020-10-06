using DrWatson
using Optim
using Distributions
using Plots
using LinearAlgebra
using Random

gr()

include(srcdir("pop_mc.jl"))
include(srcdir("gapis.jl"))
include(srcdir("lais.jl"))
include(srcdir("n3scheme.jl"))

# tgt(x, y) =
#     0.5 * pdf(MvNormal([-5.0, -5.0], [1.0 0.0; 0.0 1.0]), [x, y]) + 0.5 * pdf(MvNormal([5.0, 5.0], [1.0 0.0; 0.0 1.0]), [x, y])
# tgt(x) = tgt(x[1], x[2])

tgt(x, y) = exp(-1/(2*4^2) * (4 - 10*x - y^2)^2 - x^2 / (2*5^2) - y^2 / (2*5^2))
tgt(x) = tgt(x[1], x[2])

xsq = collect(range(-10, 10, length = 100))
ysq = xsq

fv = zeros(100, 100)
for i = 1:100
    for j = 1:100
        fv[i, j] = tgt(xsq[j], ysq[i])
    end
end

plot(xsq, ysq, fv, st = :heatmap, size = (750, 500))

pf1 = MvNormal
pf2 = MvNormal

function rcf(x::Vector, no)
    rv = [x[1], x[2]]
    return [rv, no[1]]
end


pfs = [pf1, pf2]
rcfs = [rcf, rcf]
pws = [1.0, 1.0]
parameters = [[[2.0, 2.0], [1.0 0.0; 0.0 1.0]], [[-2.0, -2.0], [1.0 0.0; 0.0 1.0]]]
nooptim = [[2], [2]]
global nv = mpmc_step(tgt, pfs, pws, parameters, 500, rcfs, nooptim, 2)

for t = 1:10
    global nv = mpmc_step(tgt, pfs, nv[2], nv[3], 500, rcfs, nooptim, 2)
end
mpc = stratified_resample(nv[1], nv[4])

# plot!(nv[1][1, :], nv[1][2, :], st = :scatter, legend = false)
# plot!(mpc[1, :], mpc[2, :], st = :scatter, legend = false)

pfs = [pf1, pf2]
locations = [[2.0, 2.0], [-2.0, -2.0]]
scales = [[1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]]
# global nv = gapis_step(tgt, pfs, locations, scales, 250, 0.1, 2)
#
# for t = 1:40
#     global nv = gapis_step(tgt, pfs, nv[2], nv[3], 250, 0.1, 2)
# end

# plot!(nv[1][1, :], nv[1][2, :], st = :scatter, legend = false)

init_mvs = [[2.0, 2.0], [-2.0, -2.0]]
fixed_params = [[1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]]

rmm = RWIS_draw_means(tgt, 2, 100, 2, init_mvs; n_adapts = 1000)

# global nv = LAIS_step(tgt, pfs, 250, 2, fixed_params, init_mvs)
# global rw = zeros(2, 500, 100)
global rmf = zeros(2, 500, 100)
global rsp = zeros(2, 500, 100)
global rmw = zeros(500, 100)
for t = 1:100
    # global rw[:,:,t] = RWIS_step(tgt, pfs, 250, 2, fixed_params, [rmm[1][t], rmm[2][t]])[1]
    global rw = RWIS_step(tgt, pfs, 250, 2, fixed_params, [rmm[1][t], rmm[2][t]])
    rmf[:,:,t] = rw[1]
    rmw[:,t] = rw[2]
    rsp[:,:,t] = stratified_resample(rw[1], rw[2])
end
rsm = stratified_resample(rw[1], rw[2])

clonk = reshape(rsp, (2, 500*100))
# thinby = randsubseq(1:(500*100), 0.02)
# cln = clonk[:, thinby]
# plot!(rsm[1, :], rsm[2, :], st = :scatter, legend = false)
plot!(clonk[1, :], clonk[2, :], st = :scatter, legend = false)
