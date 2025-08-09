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

# ─── PGA: Compute Tangent Projections ─────────────────────────────────────────
function pga_tangent_basis(M::Manifolds.AbstractManifold, points::Vector{Vector{Float64}}, k::Int)
    mean_pt = mean(M, points)
    tangent_vecs = [log(M, mean_pt, p) for p in points]
    A = hcat(tangent_vecs...)
    U, S, V = svd(A)
    return U[:, 1:k], mean_pt, tangent_vecs
end

k = 5  # top-k components
U_sig, mean_sig, tangents_sig = pga_tangent_basis(M, sig_unit, k)
U_non, mean_non, tangents_non = pga_tangent_basis(M, non_unit, k)

# ─── Compute Principal Angles Between PGA Bases ───────────────────────────────
angles = [acos(clamp(dot(U_sig[:, i], U_non[:, i]), -1.0, 1.0)) for i in 1:k]
println("Principal angles (radians) between PGA bases:", angles)

# ─── Project Points to Top-2 PGA Plane for Visualization ──────────────────────
proj2D_sig = hcat([U_sig[:, 1:2]' * v for v in tangents_sig]...)'
proj2D_non = hcat([U_non[:, 1:2]' * v for v in tangents_non]...)'

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig = Figure(resolution = (800, 400))
ax1 = Axis(fig[1, 1], title = "Significant (Top-2 PGA)", xlabel = "PC1", ylabel = "PC2")
ax2 = Axis(fig[1, 2], title = "Non-Significant (Top-2 PGA)", xlabel = "PC1", ylabel = "PC2")

scatter!(ax1, proj2D_sig[:, 1], proj2D_sig[:, 2], color = :blue, markersize = 4)
scatter!(ax2, proj2D_non[:, 1], proj2D_non[:, 2], color = :red, markersize = 4)

fig
save("pga_tangent_planes.png", fig)
