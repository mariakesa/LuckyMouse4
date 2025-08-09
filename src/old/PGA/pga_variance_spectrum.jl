using MAT
using Manifolds
using LinearAlgebra
using Statistics
using Plots

# ─── Parameters ────────────────────────────────────────────────────────────────
DATA_PATH = "/home/maria/LuckyMouse4/data/unified_neuron_data.mat"
BRAIN_REGION = "VISp"
EIGEN_CUTOFF = 1e-8
SAVE_PATH = "/home/maria/LuckyMouse4/figures/pga_variance_comparison_$(BRAIN_REGION).png"

# ─── Load data ─────────────────────────────────────────────────────────────────
file = matopen(DATA_PATH)
status = vec(read(file, "status"))
brain_area = vec(read(file, "brain_area"))
r2 = read(file, "r2")
vectors = read(file, "vector")
close(file)

# ─── Helper: compute cumulative variance spectrum ──────────────────────────────
function cumulative_variance_spectrum(vectors::Matrix{Float64})
    d = size(vectors, 2)
    M = Sphere(d - 1)
    W_unit = [normalize(view(vectors, i, :)) for i in 1:size(vectors, 1)]
    μ = mean(M, W_unit)
    tangents = [log(M, μ, w) for w in W_unit]
    T = hcat(tangents...)'
    T_centered = T .- mean(T, dims=1)
    Σ = Symmetric((T_centered' * T_centered) / (size(T_centered, 1) - 1))
    eigvals_cov = real.(eigvals(Σ))
    eigvals_filtered = filter(x -> x > EIGEN_CUTOFF, eigvals_cov)
    explained_variance = eigvals_filtered ./ sum(eigvals_filtered)
    return cumsum(sort(explained_variance; rev=true))
end

# ─── Masks ─────────────────────────────────────────────────────────────────────
mask_significant = (status .== "significant") .& (brain_area .== BRAIN_REGION)
mask_nonsignificant = (status .== "non_significant") .& (brain_area .== BRAIN_REGION)

vectors_sig = vectors[mask_significant, :]
vectors_non = vectors[mask_nonsignificant, :]

println("Loaded ", size(vectors_sig, 1), " significant and ", size(vectors_non, 1), " non-significant neurons.")

# ─── Compute cumulative variance curves ────────────────────────────────────────
curve_sig = cumulative_variance_spectrum(vectors_sig)
curve_non = cumulative_variance_spectrum(vectors_non)

# ─── Plot ──────────────────────────────────────────────────────────────────────
plot(1:length(curve_sig), curve_sig,
     label="Significant", lw=2, color=:blue)
plot!(1:length(curve_non), curve_non,
     label="Non-significant", lw=2, color=:orange)
hline!([0.9], linestyle=:dash, color=:gray, label="90% cutoff")

xlabel!("PGA component")
ylabel!("Cumulative variance explained")
title!("PGA Variance Spectrum – $(BRAIN_REGION)")

savefig(SAVE_PATH)
println("✅ Plot saved to $SAVE_PATH")
¸¸