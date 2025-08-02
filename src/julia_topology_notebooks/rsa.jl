using MAT, Manifolds, LinearAlgebra, Statistics
using CairoMakie
using StatsBase
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
using Random
Random.seed!(42)  # for reproducibility
non_indices = rand(1:size(non, 1), size(sig, 1))  # sample 831
non = non[non_indices, :]
# ─── Normalize to S^{d-1} ──────────────────────────────────────────────────────
d = size(sig, 2)
normalize_rows(X) = [normalize(view(X, i, :)) for i in 1:size(X, 1)]
sig_unit = normalize_rows(sig)
non_unit = normalize_rows(non)

# ─── Geodesic Distance Function on Sphere ─────────────────────────────────────
geo_dist(u, v) = acos(clamp(dot(u, v), -1.0, 1.0))

function pairwise_geodesic(X)
    n = length(X)
    D = zeros(n, n)
    for i in 1:n
        for j in i+1:n
            d = geo_dist(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
        end
    end
    return D
end

D_sig = pairwise_geodesic(sig_unit)
D_non = pairwise_geodesic(non_unit)
# Manual rank transform
function rankvec(x::Vector{Float64})
    sortperm(sortperm(x)) .+ 1  # convert to 1-based ranks
end


# ─── RSA: Correlation Between Dissimilarity Matrices ──────────────────────────
if size(D_sig) != size(D_non)
    println("⚠️ Warning: RDMs are not the same size, skipping RSA correlation.")
    rsa_corr = NaN
else
    mask = tril(trues(size(D_sig)), -1)
    flat_sig = vec(D_sig[mask])
    flat_non = vec(D_non[mask])  # add this to the top if not already
    rsa_corr = cor(rankvec(flat_sig), rankvec(flat_non))
end

println("RSA Spearman Correlation (Geodesic RDMs): ", rsa_corr)

# ─── Plot RDMs ────────────────────────────────────────────────────────────────
fig = Figure(size = (900, 400))
ax1 = Axis(fig[1, 1], title = "Significant RDM", xlabel = "Neuron", ylabel = "Neuron")
ax2 = Axis(fig[1, 2], title = "Non-Significant RDM", xlabel = "Neuron", ylabel = "Neuron")

heatmap!(ax1, D_sig, colormap = :viridis)
heatmap!(ax2, D_non, colormap = :viridis)

save("spherical_rdm_comparison.png", fig)
