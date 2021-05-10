__precompile__(true)

module DEM

using LinearAlgebra, LoopVectorization, Tensors, CSV, DataFrames, StaticArrays

export dem, semiaxes, saveIO, cleantensor!

"""
Axis structure
"""
struct Axis{T}
    a1::T
    a2::T
    a3::T
end
"""
Interaction tensor structure
"""
struct Tcache
    n1::Int16
    n2::Int16
    θ::Vector{Float64}
    ϕ::Vector{Float64}
    T::Array{Float64,4}
    Tv::Array{Float64,4}
    Av::Array{Float64,2}
end

# -- version 1 : one axis
@inline  semiaxes(a1::Float64,a2::Float64,a3::Float64) = [Axis(a1,a2,a3)]

# -- version 1.2 : multiple axes
function semiaxes(a10::Number,a1f::Number, Δ1::Number,
                  a20::Number,a2f::Number, Δ2::Number,
                  a30::Number,a3f::Number, Δ3::Number)

    a1 = range(a10, a1f, step=Δ1); n1 = length(a1)
    a2 = range(a20, a2f, step=Δ2); n2 = length(a2)
    a3 = range(a30, a3f, step=Δ3); n3 = length(a3)
    a1, a2, a3 = promote(a1, a2, a3)
    a  = Axis[] 
    for i1 ∈ 1:n1, i2 ∈ 1:n2, i3 ∈ 1:n3
        push!(a,Axis(a1[i1],a2[i2],a3[i3]))
    end
    return a
end

# -- version 2 : multiple axes
@inline function semiaxes(a10,a1f,a20,a2f,a30,a3f,da)
    a1 = range(a10, a1f, step=da); n1 = length(a1)
    a2 = range(a20, a2f, step=da); n2 = length(a2)
    a3 = range(a30, a3f, step=da); n3 = length(a3)

    a  = Axis[] 
    for i1 ∈ 1:n1, i2 ∈ 1:n2, i3 ∈ 1:n3
        push!(a,Axis(a1[i1],a2[i2],a3[i3]))
    end
    return a
end

# -- version 3 : given an array
@inline function semiaxes(a1::Vector{Float64},a2::Vector{Float64},a3::Vector{Float64})
    a  = Axis[] 
    for i ∈ axes(a1,1)
        push!(a,Axis(a1[i],a2[i],a3[i]))
    end
    return a
end
"""
dem(ηinc, ηmat, ϕf, dϕ, AX::Vector{Semiaxes{T}}; n1 = 100, n2 = 100)

    where ηinc is the viscosity of the inclusion, ηmat is the viscosity of the matrix, ϕf is the maximum volume fraction, dϕ is the increments of volume fraction,
    AX is the inclusion semiaxes, and n1 and n2 are the spatial discretization of θ and ϕ of the interaction tensor T  

"""
function dem(ηinc,ηmat,ϕf,dϕ,AX;n1 = 100, n2 = 100)

    # -- Make sure isotropic viscosity is Float64 to ensure type stability
    if eltype(ηinc) ≠ Float64
        ηinc = convert(Float64,ηinc)
    end
    if eltype(ηmat) ≠ Float64
        ηmat = convert(Float64,ηmat)
    end

    # -- Range of volume fractions
    ϕ           = range(dϕ,ϕf,step=dϕ)

    # -- Get some arrays which constant throughtout the whole simulation
#     n1,n2       = 100, 100    # resolution of θ and ϕ for numerical integral in interaction tensor
    cache       = getTcache(n1,n2)
    β           = βarray()    # constant array of arrays to compute 4th rank tensor inversion
        
    # -- Deviatoric viscosity tensors
    Js,Jd       = identityFourthOrder()     # Symmetric and Deviatoric 4th rank identity tensors
    Citensor    = 2 * ηinc * Jd             # Inclusion viscosity tensor
    Cmtensor    = 2 * ηmat * Jd             # Matrix viscosity tensor
    c0dem       = copy(Citensor)
    Ctemp       = copy(Cmtensor)

    # -- Voigt notation
    Cinc        = tensor2voigt(Citensor)  # Inclusion viscosity tensor in Voigts reduced notation
    Cmat        = tensor2voigt(Cmtensor)  # Matrix viscosity tensor in Voigts reduced notation
    
    # -- Output arrays
    CdemVoigt   = Array{Float64,2}[]
    outAx       = Array{Float64,1}[]
    outVol      = Float64[]

    # -- Semi axis iterations
    for i ∈ axes(AX,1)
        # - Solve DEM
        solvedem!(CdemVoigt,outAx,outVol,
            Js,Jd,Cmtensor,Citensor,AX[i],dϕ,ϕ,β,cache)        
    end
    
    return  CdemVoigt, outAx, outVol
end

"""
Parallel version
"""
function parallel_launcher(ηinc,ηmat,ϕf,dϕ,AX)
    nblk = 4*2
    sblk = floor(Int,length(a1)/nblk)
    p    = 1:sblk
    
    t1   = time()
    @sync for b ∈ 1:nblk
        # C, outAx, outVol =
        Threads.@spawn begin
            main(ηinc,ηmat,ϕf,dϕ,AX[p])
            # main(ηinc,ηmat,ϕf,dϕ,AX[p])
            p = p.+sblk
            if b == nblk-1
                p = p[1]:length(a1)
            end    
        end
        
    end
end

"""
DEM solver
"""
function solvedem!(CdemVoigt,outAx,outVol,
    Js,Jd,Cmtensor,Citensor,AX,dϕ,ϕ,β,cache)

    c0dem   = copy(Cmtensor) 
    
    # -- store results for ϕ = 0
    push!(CdemVoigt,tensor2voigt(c0dem))
    push!(outAx,[AX.a1, AX.a3,AX.a3 ])
    push!(outVol,0.0)

    ijkl    = Int8[1 6 5; 6 2 4; 5 4 3]    # voigt to tensor mapping
    A       = Array{Float64,2}(undef, 6, 6)

    # -- incremental volume fraction loop
    for ϕi ∈ ϕ        
        # -- ~time integration
        c0dem   = RK4(Js,Jd,Citensor,c0dem,β, ϕi, dϕ , AX, cache)
        # -- to Voigt
        tensor2voigt!(A, c0dem, ijkl)
        # -- store
        push!(CdemVoigt, deepcopy(A))
        push!(outAx,[AX.a1, AX.a2, AX.a3])
        push!(outVol,ϕi)
    end

end
"""
Green function
"""
# -- Green function
@inline function greenfunction(Js, Jd, Ctemp,Cinc3, β, ϕ, AX,cache)
    T           = interactiontensor(Ctemp, AX, cache)
    Ts          = symmetricinteractiontensor(T)
    dC          = SymmetricTensor{4, 3}(Cinc3 - Ctemp)
    SdC         = Ts ⊡ dC                    # Eshelby tensor := S = Jd*T*Cm = Ts*Cm 
    #=
        Add symmetric identity 4th-order tensor
        In reality it should be Ai = Id + SdC, which is however not invertible in the
        cartesian space (Lebenson et al., 1998)
    =#
    Ai          = SymmetricTensor{4, 3}(Js .+ SdC)
    # Ai          = Js + SdC
    invAi       = inv(Ai)                   # overloaded by Tensors.jl
    invAiDev    = Jd ⊡ invAi                # take deviatoric part of Ai     
    dc0dem      = dC ⊡ invAiDev             # new composite stiffness tensor    
    dc0dem      = map(x -> x * ϕ , dc0dem ) # multiply by volume fraction increment (slightly faster than normal .*)
    # return reinterpret(Float64,dc0dem)
    return SArray{NTuple{4,3}}(dc0dem)
end # end of Green function

"""
Runge-Kutta 4 
"""
# -- Runge-Kuta 4
@inline function RK4(Js, Jd, Cinc3, c0dem, β, ϕ, dϕ , AX, cache)
    ϕi      = log(1 + dϕ / (1 - ϕ))
    # c0dem   = c0dem
    Ctemp   = copy(c0dem)
    # Ctemp   = SArray{NTuple{4,3}}(c0dem)
    c0dem   = SArray{NTuple{4,3}}(c0dem)
        # (1) :=
    dC1     = greenfunction(Js, Jd, Ctemp, Cinc3, β, ϕi, AX, cache)
    Ctemp   = c0dem + dC1 / 2
        # (2) :=
    dC2     = greenfunction(Js, Jd, Ctemp, Cinc3, β, ϕi, AX, cache)
    Ctemp   = c0dem + dC2 / 2
        # (3) :=
    dC3     = greenfunction(Js, Jd, Ctemp, Cinc3, β, ϕi, AX, cache)
    Ctemp   = c0dem + dC3
        # (4) :=
    dC4     = greenfunction(Js, Jd, Ctemp, Cinc3, β, ϕi, AX, cache)
        # Final average
    c0dem   = c0dem + (dC1 + 2*dC2 + 2*dC3 + dC4) / 6
    return c0dem
end # end of Runge Kutta integration

"""
Interaction tensor and auxiliar functions
"""
function getTcache(n1,n2)
    step    = π/n1
    dθ,dϕ   = π/n1, 2π/n2
    θ       = rad2deg.(range(dθ, stop = pi,  length = n1))
    ϕ       = rad2deg.(range(dϕ, stop = 2pi, length = n2))
    T       = @MArray zeros(3,3,3,3) #fill(0.0, 3, 3, 3, 3)
    Tv      = @MArray zeros(3,3,3,3) #fill(0.0, 3, 3, 3, 3)    
    # Av      = fill(0.0, 4, 4)
    Av      = @MMatrix zeros(4,4)
    
    return Tcache(n1,n2,θ,ϕ,T,Tv,Av) 
end

@inline function interactiontensor(C, AX, cache)

    T       = cache.T # unpack structure
    Av      = @MMatrix zeros(4,4)
    invAv   = similar(Av)
    xi      = Vector{Float64}(undef, 3)
    surface = 0.0
    # xi       = 1 ./a
    for p ∈ 1:cache.n1        
        
        # to avoid redundant allocations and operations
        sinθp,cosθp = sincosd(cache.θ[p])
        
        for q ∈ 1:Int(cache.n2/2)

            # -- to avoid redundant operations          
            sinϕq,cosϕq = sincosd(cache.ϕ[q])
            
            # -- Director cosines
            xi = @SVector [sinθp*cosϕq/AX.a1, sinθp*sinϕq/AX.a2, cosθp/AX.a3]

            # -- Christoffel viscosity tensor
            Christoffel!(Av, C, xi)
            fillAv!(Av, xi)

            # -- Inverse of Christoffel tensor
            # inv4x4!(invAv,Av)
            invAv = inv(SMatrix{4,4}(Av))

            # -- Jiang, 2014, JSG
            tensorT!(T,invAv,xi,sinθp)

            surface += sinθp
        end
    end
    
    return T ./= surface
end

# -- Christoffel viscosity tensor
@inline function Christoffel!(Av,C,xi)
    @avx for t ∈ 1:3, r ∈ 1:3
        aux = zero(eltype(C))
        for u ∈ 1:3, s ∈ 1:3
            aux += C[r, s, t, u] * xi[s] * xi[u]        
        end
        Av[r, t] = aux
    end
end

@inline function tensorT!(T,invAv,xi,sinθp)
    @avx for k ∈ 1:3, i ∈ 1:3
        aux = invAv[i, k]
        for l ∈ 1:3, j ∈ 1:3
            T[i, j, k, l] += aux * xi[j] * xi[l] * sinθp
        end
    end
end

@inline function fillAv!(Av, xi)
    @avx for i ∈ 1:3
        xi0 = xi[i]
        Av[i, 4] = xi0
        Av[4, i] = xi0
    end
end

"""
Tensor operations
"""
###################################################
###   Convert C from voigt to tensor notation   ###
###################################################
@inline function voigt2tensor(A::Array{Float64,2})
    ijkl    = Int8[1 6 5; 6 2 4; 5 4 3]    # voigt to tensor mapping
    B       = Array{Float64,4}(undef, 3, 3, 3, 3)
    B       = zero(SymmetricTensor{4, 3, Float32})
    @avx for i = 1:3
        for j = 1:3
            indx = ijkl[i, j]
            for k = 1:3
                for l = 1:3
                    B[i, j, k, l] = A[indx, ijkl[k, l]]
                end
            end
        end
    end
    B = SymmetricTensor{4, 3}(B)
    return B
end ## End of function

###################################################
###   Convert C from tensor to voigt notation   ###
###################################################
@inline function tensor2voigt(A)
    ijkl    = Int8[1 6 5; 6 2 4; 5 4 3]    # voigt to tensor mapping
    B       = Array{Float64,2}(undef, 6, 6)
    @avx for l = 1:3
        for k = 1:3
            for j = 1:3
                for i = 1:3
                    B[ijkl[i, j], ijkl[k, l]] = A[i, j, k, l]
                end
            end
        end
    end
    return B
end ## End of function

@inline function tensor2voigt!(B,A,ijkl) # zero allocations version
    
    @avx for l = 1:3
        for k = 1:3
            for j = 1:3
                for i = 1:3
                    B[ijkl[i, j], ijkl[k, l]] = A[i, j, k, l]
                end
            end
        end
    end
    
end ## End of function


###################################################
### DOUBLE CONTRACTION OF TWO 4TH RANK TENSORS ###
###################################################
@inline function contraction(A::Array{Float64,4}, B::Array{Float64,4})
    C = fill(0.0, 3, 3, 3, 3)
    @avx for i = 1:3, j = 1:3, k = 1:3, l = 1:3
        aux = zero(eltype(C))
        for m = 1:3, n = 1:3
            aux += A[i, j, m, n] * B[m, n, k, l]
        end
        C[i, j, k, l] = aux
    end
    return C
end ## END OF FUNCTION

###################################################
### DOUBLE CONTRACTION OF TWO 4TH RANK TENSORS ###
###################################################
@inline function identityFourthOrder()
    # deviatoric and symmetric 4th rank identity matrices
    Js = fill(0.0, 3, 3, 3, 3)
    Jd = similar(Js)        
    @inbounds for l = 1:3, k = 1:3, j = 1:3, i = 1:3
        Js[i, j, k, l] = 0.5 * (I[i, k] * I[j, l] + I[i, l] * I[j, k])
        Jd[i, j, k, l] = Js[i, j, k, l] - I[i, j] * I[k, l] / 3
    end
    Js = SymmetricTensor{4, 3}(Js)
    Jd = SymmetricTensor{4, 3}(Jd)
    return Js, Jd
end ## END OF FUNCTION

###################################################
### DOUBLE CONTRACTION OF TWO 4TH RANK TENSORS ###
###################################################
@inline function symmetricinteractiontensor(T)
    Ts = fill(0.0, 3, 3, 3, 3)
    @avx for l = 1:3, k = 1:3, j = 1:3, i = 1:3
        Ts[i, j, k, l] +=
            0.25 * (T[i, j, k, l] + T[j, i, k, l] + T[i, j, l, k] + T[j, i, l, k])
    end
    force_symmetry!(Ts)
    Ts = SymmetricTensor{4, 3}(Ts)
    return Ts
end ## END OF FUNCTION

@inline function force_symmetry!(Ts)
    @inbounds for l = 1:3, k = 1:3, j = 1:3, i = 1:3
        Ts[j, i, k, l] = Ts[i, j, k, l]
        Ts[i, j, l, k] = Ts[i, j, k, l]
    end
end

###################################################
### Inversion of 4th rank tensor                ###
###################################################
@inline function inversefourth(X, β)
    h       = 6
    Cab     = fill(0.0, h, h)
    Cinv    = fill(0.0, 3, 3, 3, 3)
    c1, c2  = 1, 1
    # -- step 1
    @inbounds for A ∈ 1:h, B ∈ 1:h
        βA, βB = β[A], β[B]
        @fastmath for i ∈ 1:3, j ∈ 1:3, k ∈ 1:3, l ∈ 1:3
            Cab[A, B] += X[i, j, k, l] * βA[i, j] * βB[k, l]
        end
    end
    # -- step 2
    invCab = inv(Cab)
    # -- step 3
    @inbounds for A ∈ 1:h, B ∈ 1:h
        βA, βB = β[A], β[B]
        @avx for i ∈ 1:3, j ∈ 1:3, k ∈ 1:3, l ∈ 1:3
            Cinv[i, j, k, l] += invCab[A, B] * βA[i, j] * βB[k, l]
        end
    end
    Cinv = SymmetricTensor{4, 3}(Cinv)
    return Cinv
end ## END OF FUNCTION

@inline function βarray()
    # create β array of arrays
    b1 = [
        -1.0 0.0 0.0
        0.0 -1.0 0.0
        0.0 0.0 2.0
    ] ./ sqrt(6)
    b2 = [
        -1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 0.0
    ] ./ sqrt(2)
    b3 = [
        0.0 0.0 0.0
        0.0 0.0 1.0
        0.0 1.0 0.0
    ] ./ sqrt(2)
    b4 = [
        0.0 0.0 1.0
        0.0 0.0 0.0
        1.0 0.0 0.0
    ] ./ sqrt(2)
    b5 = [
        0.0 1.0 0.0
        1.0 0.0 0.0
        0.0 0.0 0
    ] / sqrt(3)
    b6 = Diagonal(fill(sqrt(3), 3))
    β  = [b1, b2, b3, b4, b5, b6]
end

"""
Save output in CSV file
"""
function saveIO(C, outAx, outVol,outName)
    n       = length(C)
    # -- Type conversions (F64 -> F32 to lighten file size)
    C       = convert(Array{Array{Float32,2},1},C)
    outAx   = convert(Array{Array{Float32,1},1},outAx)
    outVol  = convert(Array{Float32,1},outVol)
    # -- Allocations
    η11     = Vector{Float32}(undef,n)
    η22     = similar(η11)
    η33     = similar(η11)
    η44     = similar(η11)
    η55     = similar(η11)
    η66     = similar(η11)
    η12     = similar(η11)
    η13     = similar(η11)
    η23     = similar(η11)
    a1      = similar(η11)
    a2      = similar(η11)
    a3      = similar(η11)
    ϕ       = similar(η11)
    # -- Store data in vectors
    Threads.@threads for i ∈ 1:n        
        # -- Semi axis 
        a1[i]   = outAx[i][1]
        a2[i]   = outAx[i][2]
        a3[i]   = outAx[i][3]
        # -- Volume fraction 
        ϕ[i]    = outVol[i]
        # -- Viscous tensor components
        DEM     = C[i]
        η11[i]  = C[i][1,1]
        η22[i]  = C[i][2,2]
        η33[i]  = C[i][3,3]
        η44[i]  = C[i][4,4]
        η55[i]  = C[i][5,5]
        η66[i]  = C[i][6,6]
        η12[i]  = C[i][1,2]
        η13[i]  = C[i][1,3]
        η23[i]  = C[i][2,3]
    end
    # -- Make Data Frame
    df = DataFrame( nu11 = η11,
                    nu22 = η22,
                    nu33 = η33,
                    nu44 = η44,
                    nu55 = η55,
                    nu66 = η66,
                    nu12 = η12,
                    nu13 = η13,
                    nu23 = η23,
                    a1   = a1,
                    a2   = a2,
                    a3   = a3,
                    phi  = ϕ
                    )
    # -- Save data as CSV
    CSV.write(outName,df)
end

## 4 by 4 matrix functions
@inline function det4x4(A)
    detA = @inbounds @fastmath (A[1,1]*A[2,2]*A[3,3]*A[4,4] + 
            A[1,1]*A[2,3]*A[3,4]*A[4,2] + 
            A[1,1]*A[2,4]*A[3,2]*A[4,3] - 
            A[1,1]*A[2,4]*A[3,3]*A[4,2] - 
            A[1,1]*A[2,3]*A[3,2]*A[4,4] - 
            A[1,1]*A[2,2]*A[3,4]*A[4,3] - 
            A[1,2]*A[2,1]*A[3,3]*A[4,4] - 
            A[1,3]*A[2,1]*A[3,4]*A[4,2] - 
            A[1,3]*A[2,1]*A[3,2]*A[4,3] + 
            A[1,4]*A[2,1]*A[3,3]*A[4,2] + 
            A[1,3]*A[2,1]*A[3,2]*A[4,4] + 
            A[1,2]*A[2,1]*A[3,4]*A[4,3] +
            A[1,2]*A[2,3]*A[3,1]*A[4,4] + 
            A[1,3]*A[2,4]*A[3,1]*A[4,2] + 
            A[1,4]*A[2,2]*A[3,1]*A[4,3] -
            A[1,4]*A[2,3]*A[3,1]*A[4,2] - 
            A[1,3]*A[2,2]*A[3,1]*A[4,4] - 
            A[1,2]*A[2,4]*A[3,1]*A[4,3] -
            A[1,2]*A[2,3]*A[3,4]*A[4,1] - 
            A[1,3]*A[2,4]*A[3,2]*A[4,1] - 
            A[1,4]*A[2,2]*A[3,3]*A[4,1] +
            A[1,4]*A[2,3]*A[3,2]*A[4,1] +
            A[1,3]*A[2,2]*A[3,4]*A[4,1] +
            A[1,2]*A[2,4]*A[3,3]*A[4,1]
            )
end

@fastmath function adjA!(Â,A)
    Â[1,1] =@inbounds (A[2,2]*A[3,3]*A[4,4] + A[2,3]*A[3,4]*A[4,2] + A[2,4]*A[3,2]*A[4,3] -
                     A[2,4]*A[3,3]*A[4,2] - A[2,3]*A[3,2]*A[4,4] - A[2,2]*A[3,4]*A[4,3] )

    Â[1,2] =@inbounds (-A[1,2]*A[3,3]*A[4,4] - A[1,3]*A[3,4]*A[4,2] - A[1,4]*A[3,2]*A[4,3] +
                      A[1,4]*A[3,3]*A[4,2] + A[1,3]*A[3,2]*A[4,4] + A[1,2]*A[3,4]*A[4,3] )

    Â[1,3] =@inbounds (A[1,2]*A[2,3]*A[4,4] + A[1,3]*A[2,4]*A[4,2] + A[1,4]*A[2,2]*A[4,3] -
                     A[1,4]*A[2,3]*A[4,2] - A[1,3]*A[2,2]*A[4,4] - A[1,2]*A[2,4]*A[4,3] )

    Â[1,4] =@inbounds (-A[1,2]*A[2,3]*A[3,4] - A[1,3]*A[2,4]*A[3,2] - A[1,4]*A[2,2]*A[3,3] +
                      A[1,4]*A[2,3]*A[3,2] + A[1,3]*A[2,2]*A[3,4] + A[1,2]*A[2,4]*A[3,3] )

    Â[2,1] =@inbounds (-A[2,1]*A[3,3]*A[4,4] - A[2,3]*A[3,4]*A[4,1] - A[2,4]*A[3,1]*A[4,3] +
                      A[2,4]*A[3,3]*A[4,1] + A[2,3]*A[3,1]*A[4,4] + A[2,1]*A[3,4]*A[4,3] )

    Â[2,2] =@inbounds (A[1,1]*A[3,3]*A[4,4] + A[1,3]*A[3,4]*A[4,2] + A[1,4]*A[3,1]*A[4,3] -
                     A[1,4]*A[3,3]*A[4,1] - A[1,3]*A[3,1]*A[4,4] - A[1,1]*A[3,4]*A[4,3] )

    Â[2,3] =@inbounds (-A[1,1]*A[2,3]*A[4,4] - A[1,3]*A[2,4]*A[4,1] - A[1,4]*A[2,1]*A[4,3] +
                      A[1,4]*A[2,3]*A[4,1] + A[1,3]*A[2,1]*A[4,4] + A[1,1]*A[2,4]*A[4,3] )

    Â[2,4] =@inbounds (A[1,1]*A[2,3]*A[3,4] + A[1,3]*A[2,4]*A[3,1] + A[1,4]*A[2,1]*A[3,3] -
                     A[1,4]*A[2,3]*A[3,1] - A[1,3]*A[2,1]*A[3,4] - A[1,1]*A[2,4]*A[3,3] )

    Â[3,1] =@inbounds (A[2,1]*A[3,2]*A[4,4] + A[2,2]*A[3,4]*A[4,1] + A[2,4]*A[3,1]*A[4,2] -
                     A[2,4]*A[3,2]*A[4,1] - A[2,2]*A[3,1]*A[4,4] - A[2,1]*A[3,4]*A[4,2] )

    Â[3,2] =@inbounds (-A[1,1]*A[3,2]*A[4,4] - A[1,2]*A[3,4]*A[4,1] - A[1,4]*A[3,1]*A[4,2] +
                      A[1,4]*A[3,2]*A[4,1] + A[1,2]*A[3,1]*A[4,4] + A[1,1]*A[3,4]*A[4,2] )

    Â[3,3] =@inbounds (A[1,1]*A[2,2]*A[4,4] + A[1,2]*A[2,4]*A[4,1] + A[1,4]*A[2,1]*A[4,2] -
                     A[1,4]*A[2,2]*A[4,1] - A[1,2]*A[2,1]*A[4,4] - A[1,1]*A[2,4]*A[4,2] )

    Â[3,4] =@inbounds (-A[1,1]*A[2,2]*A[3,4] - A[1,2]*A[2,4]*A[3,1] - A[1,4]*A[2,1]*A[3,2] +
                      A[1,4]*A[2,2]*A[3,1] + A[1,2]*A[2,1]*A[3,4] + A[1,1]*A[2,4]*A[3,2] )

    Â[4,1] =@inbounds (-A[2,1]*A[3,2]*A[4,3] - A[2,2]*A[3,3]*A[4,1] - A[2,3]*A[3,1]*A[4,2] +
                      A[2,3]*A[3,2]*A[4,1] + A[2,2]*A[3,1]*A[4,3] + A[2,1]*A[3,3]*A[4,2] )

    Â[4,2] =@inbounds (A[1,1]*A[3,2]*A[4,3] + A[1,2]*A[3,3]*A[4,1] + A[1,3]*A[3,1]*A[4,2] -
                     A[1,3]*A[3,2]*A[4,1] - A[1,2]*A[3,1]*A[4,3] - A[1,1]*A[3,3]*A[4,2] )

    Â[4,3] =@inbounds (-A[1,1]*A[2,2]*A[4,3] - A[1,2]*A[2,3]*A[4,1] - A[1,3]*A[2,1]*A[4,2] +
                      A[1,3]*A[2,2]*A[4,1] + A[1,2]*A[2,1]*A[4,3] + A[1,1]*A[2,3]*A[4,2] )

    Â[4,4] =@inbounds (A[1,1]*A[2,2]*A[3,3] + A[1,2]*A[2,3]*A[3,1] + A[1,3]*A[2,1]*A[3,2] -
                     A[1,3]*A[2,2]*A[3,1] - A[1,2]*A[2,1]*A[3,3] - A[1,1]*A[2,3]*A[3,2] )    
end

function inv4x4!(Â,A)
    adjA!(Â,A)
    d = det4x4(A)
    @avx for j ∈ axes(A,2),i ∈ axes(A,1)
        Â[i,j] /= d
    end
end

function cleantensor!(C)
    mask1 = [1:3,4:6]
    mask2 = [4:6,1:3]
    @inbounds @simd for ll ∈ eachindex(C)
        @views C[ll][mask1[1],mask1[2]] .= 0.0
        @views C[ll][mask2[1],mask2[2]] .= 0.0
        for j ∈ 4:6, i ∈ 4:6
            if i ≠ j 
                C[ll][i,j] = 0.0
            end
        end
    end
end

end

