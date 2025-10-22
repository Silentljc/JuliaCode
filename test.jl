using TyPlot
using LinearAlgebra
include("chase.jl")

function pentadiagonal_solve(a, b, c, d, e, f)
    """
    五对角追赶法求解线性方程组 Ax = f
    其中A是五对角矩阵，使用LU分解且U矩阵主对角线元素全为1
    
    参数:
        a: 下下次对角线向量 (长度n-2)
        b: 下次对角线向量 (长度n-1)
        c: 主对角线向量 (长度n)
        d: 上次对角线向量 (长度n-1)
        e: 上上次对角线向量 (长度n-2)
        f: 右端向量 (长度n)
    
    返回:
        x: 解向量
    """
    n = length(c)
    
    # LU分解，U的主对角线设为1
    α = zeros(n)   # L的主对角线
    β = zeros(n)   # L的次对角线
    γ = zeros(n)   # L的次次对角线
    δ = zeros(n)   # U的次对角线
    ε = zeros(n)   # U的次次对角线
    
    # 初始化
    α[1] = c[1]
    if n > 1
        δ[1] = d[1] / α[1]
    end
    if n > 2
        ε[1] = e[1] / α[1]
    end
    
    if n > 1
        β[2] = b[1]
        α[2] = c[2] - β[2] * δ[1]
        if n > 2
            δ[2] = (d[2] - β[2] * ε[1]) / α[2]
        end
        if n > 3
            ε[2] = e[2] / α[2]
        end
    end
    
    # 递推计算LU分解
    for i = 3:n
        γ[i] = a[i-2]
        β[i] = b[i-1] - γ[i] * δ[i-2]
        α[i] = c[i] - γ[i] * ε[i-2] - β[i] * δ[i-1]
        
        if i < n-1
            δ[i] = (d[i] - β[i] * ε[i-1]) / α[i]
            ε[i] = e[i] / α[i]
        elseif i < n
            δ[i] = (d[i] - β[i] * ε[i-1]) / α[i]
        end
    end
    
    # 前向替代: Ly = f
    y = zeros(n)
    y[1] = f[1] / α[1]
    if n > 1
        y[2] = (f[2] - β[2] * y[1]) / α[2]
    end
    
    for i = 3:n
        y[i] = (f[i] - γ[i] * y[i-2] - β[i] * y[i-1]) / α[i]
    end
    
    # 回代: Ux = y (U主对角线为1)
    x = zeros(n)
    x[n] = y[n]
    if n > 1
        x[n-1] = y[n-1] - δ[n-1] * x[n]
    end
    
    for i = n-2:-1:1
        x[i] = y[i] - δ[i] * x[i+1] - ε[i] * x[i+2]
    end
    
    return x
end

a=ones(7)
b=ones(8)
c=ones(9)*4
d=ones(8)
e=ones(7)
f=[3;4;ones(5)*5;4;3]+ones(9)*3
x = pentadiagonal_solve(a, b, c, d, e, f)
println(x)
