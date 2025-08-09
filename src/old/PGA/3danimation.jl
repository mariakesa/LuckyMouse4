using MAT
using Manifolds
using LinearAlgebra
using Statistics
using Plots

# ─── Load both groups (edit as needed) ─────────────────────────────────────────
function load_vectors(filepath::String, group_mask)
    file = matopen(filepath)
    vectors = read(file, "vector")
    close(file)
    vectors[group_mask, :]
end

# ─── Normalize and project to tangent space ───────────────────────────────────
function pga_project(vectors::Matrix{Float64})
    M = Sphere(size(vectors, 2) - 1)
    W_unit = [normalize(view(vectors, i, :)) for i in 1:size(vectors, 1)]
    μ = mean(M, W_unit)
    coords = reduce(vcat, [log(M, μ, w)' for w in W_unit])
    return coords
end

# ─── Load group data (example masks — replace with real ones) ─────────────────
file = matopen("/home/maria/LuckyMouse4/data/unified_neuron_data.mat")
status = vec(read(file, "status"))
brain_area = vec(read(file, "brain_area"))
r2 = read(file, "r2")
close(file)

generalizing_mask = (status .== "significant") .& (brain_area .== "VISp")
nonsignificant_mask = (status .== "non_significant") .& (brain_area .== "VISp") 

vectors_generalizing = load_vectors("/home/maria/LuckyMouse4/data/unified_neuron_data.mat", generalizing_mask)
vectors_non = load_vectors("/home/maria/LuckyMouse4/data/unified_neuron_data.mat", nonsignificant_mask)

# ─── PGA projection to 3D ─────────────────────────────────────────────────────
coords_g = pga_project(vectors_generalizing)
coords_n = pga_project(vectors_non)

# ─── Keep only first 3 components (like PCA) ──────────────────────────────────
coords_g = coords_g[:, 1:3]
coords_n = coords_n[:, 1:3]

# ─── Plot and animate ─────────────────────────────────────────────────────────
gr()
plt = scatter3d(coords_g[:,1], coords_g[:,2], coords_g[:,3],
                color=:blue, label="Generalizing", alpha=0.6, markersize=2)
scatter3d!(plt, coords_n[:,1], coords_n[:,2], coords_n[:,3],
                color=:orange, label="Non-generalizing", alpha=0.6, markersize=2)

# ─── Animate rotation ─────────────────────────────────────────────────────────
elev = 20  # fixed elevation
anim = @animate for θ in 0:2:360
    plot!(plt, camera=(θ, elev))
end

gif(anim, "/home/maria/LuckyMouse4/figures/pga_rotation.gif", fps=20)

