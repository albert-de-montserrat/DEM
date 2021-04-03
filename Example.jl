using DEM
"""
Define : 
    (*) ηinc -> inclusion isotropic viscosity
    (*) ηmat -> matrix isotropic viscosity
    (*) ϕf   -> maximum volume fraction
    (*) dϕ   -> incremental volume fraction for integration
"""
ηinc,ηmat = 1e2, 1e0
ϕf        = 0.3     
dϕ        = 5e-3    

"""
Define semi-axes of the inclusion/s:

    Usage of semiaxes()
    ------------------------------------------------------------------------    
    (*) Method 1:
        semiaxes( a1,
                  a2,
                  a3 )

        a_i : (scalar) i-th principal axis    

    ------------------------------------------------------------------------
    (*) Method 2:
        semiaxes(  a1F, a1L,
                   a2F, a2L,
                   a3F, a3L,
                   dL )

        aiF : length of the First semiaxis for the i-th principal axis
        aiL : length of the Last  semiaxis for the i-th principal axis
        dL  : semiaxis increment per step

    ------------------------------------------------------------------------    
    (*) Method 2:
        semiaxes( a1,
                  a2,
                  a3 )

        a_i : array of lengths of the semiaxis for the i-th principal axis    
"""

#==================================================
# EXAMPLE OF METHOD 1 : array of axis
a1  = 1
a2  = 1
a3  = 1
AX  = semiaxes(a1, # 1st semi-axis
               a2, # 2nd semi-axis
               a3) # 3rd semi-axis

# EXAMPLE OF METHOD 2 : incremental axes
a1F, a1L = 1, 10
a2F, a2L = 1, 10
a3F, a3L = 1, 10
AX  = semiaxes(a1F,a1L, # 1st semi-axis
               a2F,a2L, # 2nd semi-axis
               a3F,a3L, # 3rd semi-axis
               1)       # incremental length
 
# EXAMPLE OF METHOD 3 : array of axes
a1  = rand(10)
a2  = rand(10)
a3  = rand(10)
AX  = semiaxes(a1, # 1st semi-axis
               a2, # 2nd semi-axis
               a3) # 3rd semi-axis
==================================================#

# EXAMPLE OF METHOD 3 : array of axes
a1  = rand(1)
a2  = rand(1)
a3  = rand(1)
AX  = semiaxes(a1, # 1st semi-axis
               a2, # 2nd semi-axis
               a3) # 3rd semi-axis
"""
Solve DEM equation
------------------------------------------------------------------------------------------------------------------
    Output :
        (*) C       := DEM viscous tensor (Array of arrays, i.e C[n][i,j] with i,j = 1,...,9, and n = number of cases)
        (*) outAX   := corresponding inclusion semiaxis (Array or arrays, outAX[1] = [a1, a2, a3])
        (*) outVol  := corresponding volume fractions   (Array)
"""
t1 = time()
C, outAx, outVol = dem(ηinc,ηmat,ϕf,dϕ,AX)
t2 = time()
println("\n Process completed after $(t2-t1) seconds \n")


"""
Save output in CSV file
"""
t1 = time();
fileName  = "Test1.csv"
saveIO( C,
        outAx, 
        outVol,
        fileName)
t2 = time();
println("\n  Output saved in: $fileName after $(t2-t1) seconds \n")
