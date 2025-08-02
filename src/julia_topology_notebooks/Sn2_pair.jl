using MAT
using Manifolds
using LinearAlgebra
using Statistics
using Plots

# ─── Load Data ───────────────────────────────────────────────────────────────
file = matopen("/home/maria/LuckyMouse4/data/unified_neuron_data.mat")
vectors = read(file, "vector")
status = vec(read(file, "status"))
brain_area = vec(read(file, "brain_area"))
close(file)

# ─── Filter Data ─────────────────────────────────────────────────────────────
X_sig = vectors[(status .== "significant") .& (brain_area .== "VISp"), :]
X_non = vectors[(status .== "non_significant") .& (brain_area .== "VISp"), :]

n_sig, d = size(X_sig)
n_non, _ = size(X_non)
M = Sphere(d - 1)

# ─── Normalize Vectors ───────────────────────────────────────────────────────
points_sig = [normalize(X_sig[i, :]) for i in 1:n_sig]
points_non = [normalize(X_non[i, :]) for i in 1:n_non]

# ─── Compute Fréchet Means ───────────────────────────────────────────────────
μ_sig = mean(M, points_sig)
μ_non = mean(M, points_non)

# ─── Project to Tangent Space and Run PGA ────────────────────────────────────
T_sig = hcat([log(M, μ_sig, p) for p in points_sig]...)
T_non = hcat([log(M, μ_non, p) for p in points_non]...)

_, S_sig, _ = svd(T_sig)
_, S_non, _ = svd(T_non)

explained_sig = S_sig.^2 ./ sum(S_sig.^2)
explained_non = S_non.^2 ./ sum(S_non.^2)

# ─── Plot Comparison ─────────────────────────────────────────────────────────
plot(
    explained_sig[1:10], label = "Significant", linewidth = 2, marker = :circle,
    title = "PGA Variance Explained (Top 10)", xlabel = "Geodesic Component",
    ylabel = "Variance Ratio"
)
plot!(explained_non[1:10], label = "Non-Significant", linewidth = 2, marker = :star)
savefig("pga_variance_comparison.png")
println("✅ Saved figure to pga_variance_comparison.png")
