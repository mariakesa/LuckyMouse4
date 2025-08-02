# analyze_neuron_geometry.jl

using MAT
using LinearAlgebra
using Manifolds
using Ripserer
using CairoMakie

# ─── Load and Filter Data ─────────────────────────────────────────────────────
file = matopen("/home/maria/LuckyMouse4/data/unified_neuron_data.mat")

neuron_idx = read(file, "neuron_idx")
r2 = read(file, "r2")
status = vec(read(file, "status"))
brain_area = vec(read(file, "brain_area"))
vectors = read(file, "vector")

close(file)

mask = (status .== "significant") .& (brain_area .== "VISp")
filtered_vectors = vectors[mask, :]
n, d = size(filtered_vectors)

println("Loaded $n vectors of dimension $d")

# ─── Normalize to Sphere S^{d-1} ──────────────────────────────────────────────
X = hcat([normalize(view(filtered_vectors, i, :)) for i in 1:n]...)  # shape (d, n)

# ─── Compute Pairwise Geodesic Distance Matrix ───────────────────────────────
W_sub = X[:, 1:300]  # 300 neurons
C = W_sub' * W_sub
D = acos.(clamp.(C, -1.0, 1.0))

# ─── Run Persistent Homology ─────────────────────────────────────────────────
println("Running Ripserer...")
result = ripserer(D; dim_max=2, threshold=π)

# ─── Plot Persistence Diagrams ───────────────────────────────────────────────
fig = Figure(size = (800, 300))
for (dim, dgms) in enumerate(result)
    ax = Axis(fig[1, dim]; title = "H$(dim)", xlabel = "Birth", ylabel = "Death")
    scatter!(ax, birth.(dgms), death.(dgms); markersize = 6)
    lines!(ax, [0, π], [0, π]; color = :gray, linestyle = :dot)
end

save("persistence_diagrams.png", fig)
println("✅ Saved diagram to persistence_diagrams.png")
