using MAT
using Manifolds
using LinearAlgebra
using Statistics
using Plots

# ─── Parameters ────────────────────────────────────────────────────────────────
DATA_PATH = "/home/maria/LuckyMouse4/data/unified_neuron_data.mat"
BRAIN_REGION = "VISp"
MIN_NORM = 1e-6
EIGEN_CUTOFF = 1e-8  # Ignore eigenvalues below this threshold

# ─── Load and Filter Data ─────────────────────────────────────────────────────
file = matopen(DATA_PATH)

neuron_idx = read(file, "neuron_idx")
r2 = read(file, "r2")
status = vec(read(file, "status"))
brain_area = vec(read(file, "brain_area"))
vectors = read(file, "vector")

close(file)

mask = (status .== "non_significant") .& (brain_area .== BRAIN_REGION)
selected_vectors = vectors[mask, :]

println("Loaded $(size(selected_vectors, 1)) neurons with significant selectivity in $BRAIN_REGION.")

# Filter low-norm neurons
norms = [norm(view(selected_vectors, i, :)) for i in 1:size(selected_vectors, 1)]
mask_norm = norms .> MIN_NORM
filtered_vectors = selected_vectors[mask_norm, :]
println("Filtered out $(sum(.!mask_norm)) neurons with near-zero weights.")
println("Remaining neurons: $(size(filtered_vectors, 1))")

# ─── Normalize onto unit sphere ───────────────────────────────────────────────
d = size(filtered_vectors, 2)
M = Sphere(d - 1)
W_unit = [normalize(view(filtered_vectors, i, :)) for i in 1:size(filtered_vectors, 1)]

# ─── Compute tangent vectors ──────────────────────────────────────────────────
μ = mean(M, W_unit)
tangents = [log(M, μ, w) for w in W_unit]
T = hcat(tangents...)'   # neurons x features

# ─── Stable covariance ────────────────────────────────────────────────────────
T_centered = T .- mean(T, dims=1)
Σ = Symmetric((T_centered' * T_centered) / (size(T, 1) - 1))
eigvals_Σ = real.(eigvals(Σ))

# Filter meaningful eigenvalues
eigvals_nonzero = filter(x -> x > EIGEN_CUTOFF, eigvals_Σ)

if length(eigvals_nonzero) >= 2
    anisotropy_ratio = maximum(eigvals_nonzero) / minimum(eigvals_nonzero)
else
    anisotropy_ratio = 1.0
end

println("Eigenvalues kept ( >", EIGEN_CUTOFF, " ): ", eigvals_nonzero)
println("Anisotropy ratio (max/min): ", anisotropy_ratio)

if anisotropy_ratio < 1.5
    println("⚠️ The logistic regression weights are nearly isotropic → likely noise-dominated.")
elseif anisotropy_ratio >= 1.5 && anisotropy_ratio < 3
    println("ℹ️ Moderate anisotropy → weak directional preference in weights.")
else
    println("✅ Strong anisotropy detected (after filtering numerical noise).")
end

# ─── PGA visualization (renamed to avoid coords conflict) ─────────────────────
pga_points = reduce(vcat, [log(M, μ, w)' for w in W_unit])
if size(pga_points, 2) >= 2
    scatter(pga_points[:,1], pga_points[:,2],
        alpha=0.6,
        xlabel="PGA 1", ylabel="PGA 2",
        title="PGA projection of neuron weight vectors")
end
