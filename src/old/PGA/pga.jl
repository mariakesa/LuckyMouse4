using MAT
using Manifolds
using LinearAlgebra
using Statistics

# ─── Load and Filter Data ─────────────────────────────────────────────────────
file = matopen("/home/maria/LuckyMouse4/data/unified_neuron_data.mat")

neuron_idx = read(file, "neuron_idx")
r2 = read(file, "r2")
status = vec(read(file, "status"))
brain_area = vec(read(file, "brain_area"))
vectors = read(file, "vector")

close(file)

mask = (status .== "non_significant") .& (brain_area .== "VISp")
significant_visp_vectors = vectors[mask, :]

println("Loaded $(size(significant_visp_vectors, 1)) neurons with significant selectivity in VISp.")

# ─── Normalize Vectors onto the Sphere S^{d-1} ────────────────────────────────
d = size(significant_visp_vectors, 2)
M = Sphere(d - 1)

W_unit = [normalize(view(significant_visp_vectors, i, :)) for i in 1:size(significant_visp_vectors, 1)]

# ─── Compute PGA ──────────────────────────────────────────────────────────────
μ = mean(M, W_unit)
tangents = [log(M, μ, w) for w in W_unit]
T = hcat(tangents...)'   # (neurons x features)

# ─── Covariance and isotropy analysis ─────────────────────────────────────────
Σ = cov(T)
eigvals_Σ = eigvals(Σ)
anisotropy_ratio = maximum(eigvals_Σ) / minimum(eigvals_Σ)

println("Eigenvalues of covariance matrix:")
println(eigvals_Σ)
println("Anisotropy ratio (max/min): ", anisotropy_ratio)

if anisotropy_ratio ≈ 1
    println("⚠️ The logistic regression weights are nearly isotropic → likely noise-dominated (normal distribution).")
elseif anisotropy_ratio > 3
    println("✅ Strong anisotropy detected → likely presence of structured selectivity directions.")
else
    println("ℹ️ Moderate anisotropy → weak directional preference in weights.")
end
