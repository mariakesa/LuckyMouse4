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

# ─── PGA: Shared Tangent Space at Mean of Significant Group ───────────────────
function pga_tangent_basis(M::Manifolds.AbstractManifold, points::Vector{Vector{Float64}}, k::Int)
    mean_pt = mean(M, points)
    tangent_vecs = [log(M, mean_pt, p) for p in points]
    A = hcat(tangent_vecs...)
    U, S, V = svd(A)
    return U[:, 1:k], mean_pt, tangent_vecs
end

k = 5  # top-k components
U_sig, mean_sig, tangents_sig = pga_tangent_basis(M, sig_unit, k)
tangents_non = [log(M, mean_sig, p) for p in non_unit]  # use shared mean

# ─── Project Both Groups onto Same PGA Plane ──────────────────────────────────
proj2D_sig = hcat([U_sig[:, 1:2]' * v for v in tangents_sig]...)'
proj2D_non = hcat([U_sig[:, 1:2]' * v for v in tangents_non]...)'

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig = Figure(resolution = (600, 600))
ax = Axis(fig[1, 1], title = "Shared PGA Space (Top-2)", xlabel = "PC1", ylabel = "PC2")

scatter!(ax, proj2D_sig[:, 1], proj2D_sig[:, 2], color = :blue, markersize = 4, label = "Significant")
scatter!(ax, proj2D_non[:, 1], proj2D_non[:, 2], color = :red, markersize = 4, label = "Non-Significant")
axislegend(ax, position = :rb)

fig
save("pga_shared_tangent.png", fig)
