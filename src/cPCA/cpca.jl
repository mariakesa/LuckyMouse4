using MAT, Manifolds, LinearAlgebra, Statistics, CairoMakie

# ─── Load Data ────────────────────────────────────────────────────────────────
file = matopen("/home/maria/LuckyMouse4/data/unified_neuron_data.mat")
vectors = read(file, "vector")
status = vec(read(file, "status"))
brain_area = vec(read(file, "brain_area"))
close(file)

# ─── Select Groups ────────────────────────────────────────────────────────────
fg_mask = (status .== "significant") .& ((brain_area .== "VISam") .| (brain_area .== "VISp"))
bg_mask = (status .== "non_significant") .& (brain_area .== "VISp")

fg = vectors[fg_mask, :]
bg = vectors[bg_mask, :]

fg_labels = brain_area[fg_mask]  # to color VISam vs VISp separately

# ─── Normalize to S^{d-1} ──────────────────────────────────────────────────────
d = size(fg, 2)
M = Manifolds.Sphere(d - 1)
normalize_rows(X) = [normalize(view(X, i, :)) for i in 1:size(X, 1)]
fg_unit = normalize_rows(fg)
bg_unit = normalize_rows(bg)

# ─── Contrastive PGA ──────────────────────────────────────────────────────────
function contrastive_pga(fg::Vector{Vector{Float64}}, bg::Vector{Vector{Float64}}, α::Float64, k::Int)
    mean_fg = mean(M, fg)
    tangents_fg = [log(M, mean_fg, p) for p in fg]
    tangents_bg = [log(M, mean_fg, p) for p in bg]

    X_fg = hcat(tangents_fg...)
    X_bg = hcat(tangents_bg...)

    C_fg = X_fg * X_fg' / size(X_fg, 2)
    C_bg = X_bg * X_bg' / size(X_bg, 2)
    C_contrast = C_fg - α * C_bg

    evals, evecs = eigen(Symmetric(C_contrast))
    idxs = sortperm(evals; rev=true)
    U = evecs[:, idxs[1:k]]
    return U, mean_fg, tangents_fg, tangents_bg
end

# ─── Run Contrastive PGA ──────────────────────────────────────────────────────
α = 0.1
k = 5
U_contrastive, mean_fg, tangents_fg, tangents_bg = contrastive_pga(fg_unit, bg_unit, α, k)

# ─── Project to Top-2 Contrastive Geodesics ────────────────────────────────────
proj2D_fg = hcat([U_contrastive[:, 1:2]' * v for v in tangents_fg]...)'

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig = Figure(size = (600, 600))
ax = Axis(fig[1, 1], title = "Contrastive PGA (α = $α)", xlabel = "cPC1", ylabel = "cPC2")

scatter!(ax, proj2D_fg[fg_labels .== "VISam", 1], proj2D_fg[fg_labels .== "VISam", 2], 
        color = :orange, markersize = 4, label = "VISam")
scatter!(ax, proj2D_fg[fg_labels .== "VISp", 1], proj2D_fg[fg_labels .== "VISp", 2], 
        color = :blue, markersize = 4, label = "VISp")
axislegend(ax, position = :rb)

fig
save("cpga_visam_visp.png", fig)
