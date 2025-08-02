using MAT, Manifolds, LinearAlgebra, Statistics, CairoMakie

# ─── Load Data ────────────────────────────────────────────────────────────────
file = matopen("/home/maria/LuckyMouse4/data/unified_neuron_data.mat")
vectors = read(file, "vector")
status = vec(read(file, "status"))
brain_area = vec(read(file, "brain_area"))
close(file)

# ─── Select VISp Neurons ──────────────────────────────────────────────────────
sig_mask = (status .== "significant") .& (brain_area .== "VISp")
non_mask = (status .== "non_significant") .& (brain_area .== "VISp")

sig = vectors[sig_mask, :]
non = vectors[non_mask, :]

# ─── Normalize to S^{d-1} ──────────────────────────────────────────────────────
d = size(sig, 2)
M = Manifolds.Sphere(d - 1)
normalize_rows(X) = [normalize(view(X, i, :)) for i in 1:size(X, 1)]
sig_unit = normalize_rows(sig)
non_unit = normalize_rows(non)

# ─── Shared Tangent Point ─────────────────────────────────────────────────────
mu = mean(M, sig_unit)  # or combine sig + non for more neutral reference
sig_log = [log(M, mu, p) for p in sig_unit]
non_log = [log(M, mu, p) for p in non_unit]

# ─── Compute Covariances ──────────────────────────────────────────────────────
A_sig = hcat(sig_log...)
A_non = hcat(non_log...)
C_sig = cov(A_sig')  # rows = observations
C_non = cov(A_non')

# ─── Contrastive PGA ──────────────────────────────────────────────────────────
function contrastive_pga(C_target, C_bg, alpha::Float64, k::Int)
    C_diff = C_target - alpha * C_bg
    evals, evecs = eigen(Symmetric(C_diff))
    idx = sortperm(evals; rev=true)[1:k]
    return evecs[:, idx], evals[idx]
end

# ─── Sweep Over Alpha ─────────────────────────────────────────────────────────
alphas = [0.0, 0.1, 0.25, 0.5, 1.0]
k = 3

fig = Figure(resolution=(900, 300))
for (i, alpha) in enumerate(alphas)
    U_c, λ = contrastive_pga(C_sig, C_non, alpha, k)
    proj_sig = hcat([U_c[:, 1:2]' * v for v in sig_log]...)'
    proj_non = hcat([U_c[:, 1:2]' * v for v in non_log]...)'

    ax = Axis(fig[1, i], title = "α = $alpha", xlabel = "cPC1", ylabel = "cPC2")
    scatter!(ax, proj_non[:, 1], proj_non[:, 2], color = (:red, 0.4), markersize = 4)
    scatter!(ax, proj_sig[:, 1], proj_sig[:, 2], color = (:blue, 0.6), markersize = 4)
end
fig
save("cpga_visp_projection_sweep.png", fig)

# ─── Optional: Compare with Standard PGA Cosine Overlap ───────────────────────
using LinearAlgebra: dot, norm
α = 1.0  # or 0.5 or any Float64 you want to try
W_c = contrastive_pga(log_sig, log_non, α, k)
U_c, _ = contrastive_pga(C_sig, C_non, 1.0, k)
cos_sim = [abs(dot(U_pga[:, i], U_c[:, i])) / (norm(U_pga[:, i]) * norm(U_c[:, i])) for i in 1:k]
println("Cosine similarity between PGA and cPGA basis (alpha=1):", cos_sim)
