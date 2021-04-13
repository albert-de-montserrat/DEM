### A Pluto.jl notebook ###
# v0.14.1

using InteractiveUtils

# ╔═╡ b6b55a64-08d5-11eb-0658-ada418d3a58a
using DEM, Markdown

# ╔═╡ 91b4a99b-ebf0-4900-85dd-07514c968a07
md"# Material properties 
    ηinc -> inclusion isotropic viscosity
    ηmat -> matrix isotropic viscosity
    ϕf   -> maximum volume fraction
    dϕ   -> incremental volume fraction for integration"

# ╔═╡ cb317c9a-08d5-11eb-360d-a1346cc5cedb
begin
	ηinc,ηmat = 1.0, 1e3;  # inclusion and matrix isotropic viscosities
	ϕf        = 0.20;      # maximum volume fraction
	dϕ        = 0.005;     # incremental volume fraction
end

# ╔═╡ a7feb54b-9b3a-4c27-8865-65c22c474efa
md"# Define set of semi-axes"

# ╔═╡ 411c6e60-06b0-4ab3-a992-58c62181f3a3
md"Method 1:
        semiaxes( a1<:Real,
                  a2<:Real,
                  a3<:Real )

        a_i : (scalar) i-th principal axis"

# ╔═╡ 5d67a3ac-2a7a-40c2-b8e4-1df1ec2cd318
md"Method 2:
        semiaxes(  a1F<:Real, a1L<:Real,
                   a2F<:Real, a2L<:Real,
                   a3F<:Real, a3L<:Real,
                   dL<:Real )

        aᵢF : length of the First semiaxis for the i-th principal axis
        aᵢL : length of the Last  semiaxis for the i-th principal axis
        dL  : semiaxis increment per step"

# ╔═╡ 51228092-9271-43f5-a90a-508ae0403bb8
md"Method 3:
        semiaxes( a1::Vector{Real},
                  a2::Vector{Real},
                  a3::Vector{Real} )

        a_i : array containing the lengths of the semiaxis for the i-th principal axis  "

# ╔═╡ bd81b598-13ab-4f95-b32e-72ad0052187e
md"Example method 1:"

# ╔═╡ ba5f7d6c-096f-11eb-106b-75fed594ad68
begin
	a1 = 1e3 	# 1st semi-axis
	a2 = 50.0 	# 2nd semi-axis
	a3 = 1.0    # 3rd semi-axis
	AX  = semiaxes(a1, # 1st semi-axis
               a2, # 2nd semi-axis
               a3) # 3rd semi-axisend
end

# ╔═╡ 9859c162-79f6-4743-a54f-a7418acaa884
md"# Solve DEM equation

C, outAx, outVol = dem(ηinc<:Real, ηmat<:Real, ϕf<:Float64, dϕ::Float64, AX::Vector{Semiaxes{T}}; n1::Int=100, n2::Int=100)

	Input:
		ηinc → inclusion viscosity
		ηmat → matrix viscosity 
		ϕf → volume fraction
		dϕ → volume fraction increments
		AX → Array of inclusion semiaxes
		n1 → θ spatial discretization for integration of interaction tensor T
		n2 → ϕ spatial discretization for integration of interaction tensor T

	Output:
        C → viscous tensor (Array of arrays, C[n][i,j] with i,j = 1,...,9, n = #cases)
        outAX → array ofinclusion semiaxis (outAX[n] = [a1, a2, a3])
        outVol → volume fractions (Array)

"

# ╔═╡ 3fa80934-08d6-11eb-0b12-574e70931de1
C, outAx, outVol = dem(ηinc,ηmat,ϕf,dϕ,AX);

# ╔═╡ 44dd0b30-4b48-4dc6-a76c-a437d9ee4b0d
cleantensor!(C) # remove floating point errors

# ╔═╡ 51c906ba-096f-11eb-1108-31010cd0c77b
ϕ = 0.01; # show tensor at this volume fraction

# ╔═╡ 771064d6-08d6-11eb-105b-6dcfc5d3179c
ivol = findall(x->x==ϕ, outVol)[1];

# ╔═╡ 369ac2df-c296-4a27-b17c-3d23c7ebb45e
C[3]

# ╔═╡ 0c961cab-029b-43e6-b256-ec4c7cfb230c
pwd()

# ╔═╡ Cell order:
# ╠═b6b55a64-08d5-11eb-0658-ada418d3a58a
# ╟─91b4a99b-ebf0-4900-85dd-07514c968a07
# ╠═cb317c9a-08d5-11eb-360d-a1346cc5cedb
# ╟─a7feb54b-9b3a-4c27-8865-65c22c474efa
# ╟─411c6e60-06b0-4ab3-a992-58c62181f3a3
# ╟─5d67a3ac-2a7a-40c2-b8e4-1df1ec2cd318
# ╟─51228092-9271-43f5-a90a-508ae0403bb8
# ╟─bd81b598-13ab-4f95-b32e-72ad0052187e
# ╠═ba5f7d6c-096f-11eb-106b-75fed594ad68
# ╟─9859c162-79f6-4743-a54f-a7418acaa884
# ╠═3fa80934-08d6-11eb-0b12-574e70931de1
# ╠═44dd0b30-4b48-4dc6-a76c-a437d9ee4b0d
# ╠═51c906ba-096f-11eb-1108-31010cd0c77b
# ╠═771064d6-08d6-11eb-105b-6dcfc5d3179c
# ╠═369ac2df-c296-4a27-b17c-3d23c7ebb45e
# ╠═0c961cab-029b-43e6-b256-ec4c7cfb230c
