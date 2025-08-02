using Manifolds
using LinearAlgebra
using Statistics
using Plots
using MAT

# ─── Load Vectors from .mat File ─────────────────────────────────────────────
file = matopen("/home/maria/LuckyMouse4/data/unified_neuron_data.mat")
vectors = read(file, "vector")
status = vec(read(file, "status"))
brain_area = vec(read(file, "brain_area"))
close(file)

# ─── Filter VISp Significant Neurons ─────────────────────────────────────────
mask = (status .== "significant") .& (brain_area .== "VISp")
X = vectors[mask, :]  # size (n, d)
n, d = size(X)

# ─── Normalize to the Sphere S^{d-1} ─────────────────────────────────────────
M = Sphere(d - 1)
points = [normalize(X[i, :]) for i in 1:n]

# ─── Compute Pairwise Geodesic Distances ─────────────────────────────────────
D = [distance(M, points[i], points[j]) for i in 1:n, j in 1:n]
println("Mean geodesic distance: ", mean(D))

# ─── Compute Fréchet Mean ────────────────────────────────────────────────────
μ = mean(M, points)
println("Computed Fréchet mean on S^$(d-1)")

# ─── Log Map to Tangent Space at Mean ────────────────────────────────────────
log_vecs = [log(M, μ, p) for p in points]  # tangent vectors in T_μM

# ─── Principal Geodesic Analysis (PGA) ───────────────────────────────────────
# Like PCA in tangent space
T = hcat(log_vecs...)  # tangent vectors as columns
U, S, V = svd(T)
explained_variance = S.^2 ./ sum(S.^2)

# ─── Plot Variance Explained by PGA ──────────────────────────────────────────
plot(
    explained_variance[1:10],
    seriestype = :bar,
    legend = false,
    title = "Principal Geodesic Variance Explained",
    xlabel = "Component",
    ylabel = "Variance Ratio",
    size = (600, 400)
)
savefig("geodesic_variance.png")
println("✅ Saved geodesic variance plot to geodesic_variance.png")
