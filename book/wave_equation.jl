# 波方程

using TyPlot
using LinearAlgebra
using SpecialFunctions 
using Printf
using ProgressMeter

include("../solver/chase.jl")

function solve_wave_equation()
    start_time = time()

    γ = 1.9
    L = 1
    T = 1
    # m = fill(20000, 2)
    # n = [640,1280]
    m = [20,40]
    n = m.^2

    error_inf = zeros(length(m))
    Norm = zeros(length(m)-1)

    for p in 1:length(m)
        println("\n正在处理第 $p 个网格 (m=$(m[p]), n=$(n[p]))")
        h = L/m[p]
        τ = T/n[p]
        x = 0:h:L
        t = 0:τ:T

        η = (τ^(γ-1))*gamma(3-γ)

        u = zeros(m[p]+1, n[p]+1)

        f(x,t) = exp(x)*t^4*(gamma(5+γ)/24-t^γ)

        # ut(x,0)
        ψ(x) = 0


        # 边界值
        u[:,1] = zeros(m[p]+1)
        u[1,:] = t.^(4+γ)
        u[m[p]+1,:] = exp(1).*t.^(4+γ)

        # 精确解
        Accurate = zeros(m[p]+1, n[p]+1);
        for i in 1:m[p]+1
            for k in 1:n[p]+1
                Accurate[i,k] = exp(x[i])*t[k]^(4+γ)
            end
        end

        b_γ(l) = (l+1)^(2-γ)-l^(2-γ) 

        a = (-1/(2*h^2)).*ones(m[p]-2)
        b = (b_γ(0)/(η*τ) + 1/h^2).*ones(m[p]-1)

        @showprogress barlen=50 dt=1 "时间步迭代进度" for k in 2:n[p]+1
            # 创建右端向量
            right_vector = zeros(m[p]-1)
            for i in 1:m[p]-1
                Σ = 0
                if k!=2
                    for l in 1:k-2 
                        Σ += (b_γ(k-1-l-1)-b_γ(k-1-l))*(u[i+1,l+1]-u[i+1,l])/τ
                    end
                end

                right_vector[i] = (1/η)*(Σ + b_γ(k-1-1)*ψ(x[i+1])) + f(x[i+1], t[k]-τ/2)+
                    (1/(2*h^2))*u[i,k-1]+(b_γ(0)/(η*τ)-1/h^2)*u[i+1,k-1]+(1/(2*h^2))*u[i+2,k-1]
            end

            # 添加边界贡献
            right_vector[1] -= (-1/(2*h^2))*u[1,k]
            right_vector[m[p]-1] -= (-1/(2*h^2))*u[m[p]+1,k]

            # 求解三对角线性方程组
            u[2:m[p],k] = chase(a, b, a, right_vector)

        end

      
        error = abs.(u[:,:] - Accurate[:,:])
        error_inf[p] = maximum(error)
        
        # 绘图
        # figure(p)
        # X = repeat(x,1, length(t))
        # T_grid = repeat(t', length(x),1)
        
        # subplot(1,3,1)
        # surf(X, T_grid, Accurate[:,:])
        # xlabel("x"); ylabel("t"); zlabel("Accurate")
        # title("精确解")
        # grid(true)
        
        # subplot(1,3,2)
        # surf(X, T_grid, u[:,:])
        # xlabel("x"); ylabel("t"); zlabel("Numerical")
        # title("数值解")
        # grid(true)
        
        # subplot(1,3,3)
        # surf(X, T_grid, error)
        # xlabel("x"); ylabel("t"); zlabel("error")
        # title("误差")
        # grid(true)

    end

    # 计算收敛阶
    for k = 2:length(m)
        X = error_inf[k-1]/error_inf[k]
        Norm[k-1] = log2(X)
    end

    # figure(length(n)+1)
    # plot(1:length(n)-1, Norm, "-b^")
    # xlabel("序号"); ylabel("误差阶数")
    # title("格式误差阶")
    # grid(true)

    println("\n误差与收敛阶表：")
    @printf("%-10s %-10s %-15s %-10s\n", "m", "n", "error_inf", "收敛阶")
    for i in 1:length(m)
        if i == 1
            @printf("%-10d %-10d %-15.6e %-10s\n", m[i], n[i], error_inf[i], "-")
        else
            @printf("%-10d %-10d %-15.6e %-10.4f\n", m[i], n[i], error_inf[i], Norm[i-1])
        end
    end

    elapsed_time = time() - start_time
    minutes = floor(Int, elapsed_time / 60)
    seconds = round(Int, elapsed_time % 60)
    
    if minutes > 0
        println("\n程序共运行$(minutes)分$(seconds)秒")
    else
        println("\n程序共运行$(seconds)秒")
    end

end

solve_wave_equation()