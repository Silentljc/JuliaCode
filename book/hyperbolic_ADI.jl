# 双曲ADI

using TyPlot
using LinearAlgebra
using Printf

include("../chase.jl")

function solve_adi_problem()
    start_time = time()
    M = [10,20,40,80,160]  # 空间网格数量
    N = M  # 时间网格数量
    error_inf = zeros(length(M))
    Norm = zeros(length(M)-1)

    for p in 1:length(M)
        h = 1/M[p]  # 空间步长
        τ = 1/N[p]  # 时间步长 (使用LaTeX表示的tau)
        x = 0:h:1
        y = 0:h:1
        t = 0:τ:1
        
        Numerical = zeros(M[p]+1, M[p]+1, N[p]+1)  # u
        numerical = zeros(M[p]+1, M[p]-1)  # u*

        r = τ^2/(h^2)
        α = -r/2  # 使用LaTeX表示的alpha
        β = 1+r   # 使用LaTeX表示的beta
        
        # 构造三对角矩阵系数
        a = α * ones(M[p]-2)
        b = β * ones(M[p]-1)
        c = α * ones(M[p]-2)

        # 设置初值
        for i in 1:M[p]+1
            for j in 1:M[p]+1
                Numerical[i,j,1] = exp(1/2*(x[i]+y[j]))  # 初值
            end
        end
        
        # 设置边界条件
        for j in 1:M[p]+1
            for k in 1:N[p]+1
                Numerical[1,j,k] = exp(1/2*y[j]-t[k])  # u(0,y,t)
            end
        end
        
        for j in 1:M[p]+1
            for k in 1:N[p]+1
                Numerical[M[p]+1,j,k] = exp(1/2*(1+y[j])-t[k])  # u(1,y,t)
            end
        end
        
        for i in 1:M[p]+1
            for k in 1:N[p]+1
                Numerical[i,1,k] = exp(1/2*x[i]-t[k])  # u(x,0,t)
            end
        end
        
        for i in 1:M[p]+1
            for k in 1:N[p]+1
                Numerical[i,M[p]+1,k] = exp(1/2*(1+x[i])-t[k])  # u(x,1,t)
            end
        end
        
        # 定义函数
        f_func = (x,y,t) -> 1/2*exp(1/2*(x+y)-t)
        fun = (x,y,t) -> exp(1/2*(x+y)-t)
        ψ = (x,y) -> -exp(1/2*(x+y))  # 使用LaTeX表示的psi
        ρ = (x,y) -> -exp(1/2*(x+y))  # 使用LaTeX表示的rho
        
        # 计算精确解
        Accurate = zeros(M[p]+1, M[p]+1, N[p]+1)
        for i in 1:M[p]+1
            for j in 1:M[p]+1
                for k in 1:N[p]+1
                    Accurate[i,j,k] = fun(x[i], y[j], t[k])
                end
            end
        end

        # 计算Numerical（x,y,1）=u(i,j,1)
        for j in 1:M[p]-1  # 固定j
            numerical[1,j] = α*Numerical[1,j,2] +
                β*Numerical[1,j+1,2] +
                α*Numerical[1,j+2,2]  # u*0j1
            numerical[M[p]+1,j] = α*Numerical[M[p]+1,j,2] +
                β*Numerical[M[p]+1,j+1,2] +
                α*Numerical[M[p]+1,j+2,2]  # u*mj1
            
            # 循环生成右端列向量
            numerical_right_vector = zeros(M[p]-1)
            for i in 1:M[p]-1
                numerical_right_vector[i] = Numerical[i+1,j+1,1] + τ*ψ(x[i+1],y[j+1]) - τ^3*ρ(x[i+1],y[j+1])/3 +
                    τ^2*f_func(x[i+1],y[j+1],t[2])/2
            end
            
            # 添加边界贡献
            numerical_right_vector[1] -= α * numerical[1,j]
            numerical_right_vector[M[p]-1] -= α * numerical[M[p]+1,j]
            
            numerical[2:M[p],j] = chase(a, b, c, numerical_right_vector)
        end
        
        for i in 1:M[p]-1  # 固定i
            Numerical_right_vector = zeros(M[p]-1)
            for j in 1:M[p]-1
                Numerical_right_vector[j] = numerical[i+1,j]
            end
            
            Numerical_right_vector[1] -= α * Numerical[i+1,1,2]
            Numerical_right_vector[M[p]-1] -= α * Numerical[i+1,M[p]+1,2]
            
            Numerical[i+1,2:M[p],2] = chase(a, b, c, Numerical_right_vector)
        end

        # 核心部分
        for k in 2:N[p]
            for j in 1:M[p]-1  # 固定j
                numerical[1,j] = α*(Numerical[1,j,k+1]+Numerical[1,j,k-1])/2 +
                    β*(Numerical[1,j+1,k+1]+Numerical[1,j+1,k-1])/2 +
                    α*(Numerical[1,j+2,k+1]+Numerical[1,j+2,k-1])/2  # u*0j
                numerical[M[p]+1,j] = α*(Numerical[M[p]+1,j,k+1]+Numerical[M[p]+1,j,k-1])/2 +
                    β*(Numerical[M[p]+1,j+1,k+1]+Numerical[M[p]+1,j+1,k-1])/2 +
                    α*(Numerical[M[p]+1,j+2,k+1]+Numerical[M[p]+1,j+2,k-1])/2  # u*mj
                
                # 循环生成右端列向量
                numerical_right_vector = zeros(M[p]-1)
                for i in 1:M[p]-1
                    # numerical_right_vector[i] = 2*Numerical[i+1,j+1,k] + f_func(x[i+1],y[j+1],t[k])*τ^2 +
                    #     (r/2)*(Numerical[i,j+1,k-1] + Numerical[i+2,j+1,k-1]) +
                    #     (r/2)*(Numerical[i+1,j,k-1] + Numerical[i+1,j+2,k-1]) - 
                    #     (1+r+r)*Numerical[i+1,j+1,k-1]
                    numerical_right_vector[i] = Numerical[i+1,j+1,k]+τ^2/2*f_func(x[i+1],y[j+1],t[k])
                end
                
                # 添加边界贡献
                numerical_right_vector[1] -= α * numerical[1,j]
                numerical_right_vector[M[p]-1] -= α * numerical[M[p]+1,j]
                
                numerical[2:M[p],j] = chase(a, b, c, numerical_right_vector)
            end
            
            for i in 1:M[p]-1  # 固定i
                temp1 = (Numerical[i+1,1,k+1]+Numerical[i+1,1,k-1])/2
                temp2 = (Numerical[i+1,M[p]+1,k+1]+Numerical[i+1,M[p]+1,k-1])/2
                Numerical_right_vector = zeros(M[p]-1)
                for j in 1:M[p]-1
                    Numerical_right_vector[j] = numerical[i+1,j]
                end
                
                # Numerical_right_vector[1] -= α * Numerical[i+1,1,k+1]
                # Numerical_right_vector[M[p]-1] -= α * Numerical[i+1,M[p]+1,k+1]
                Numerical_right_vector[1] -= α * temp1
                Numerical_right_vector[M[p]-1] -= α * temp2
                
                Numerical[i+1,2:M[p],k+1] = 2*chase(a, b, c, Numerical_right_vector)-Numerical[i+1,2:M[p],k-1]
            end
        end

        error = abs.(Numerical[:,:,end] - Accurate[:,:,end])
        error_inf[p] = maximum(error)
        
        # 绘图
        # figure(p)
        # X = repeat(y', length(x), 1)
        # Y = repeat(x, 1, length(y))
        
        # subplot(1,3,1)
        # surf(X, Y, Accurate[:,:,end])
        # xlabel("x"); ylabel("y"); zlabel("Accurate")
        # title("精确解")
        # grid(true)
        
        # subplot(1,3,2)
        # surf(X, Y, Numerical[:,:,end])
        # xlabel("x"); ylabel("y"); zlabel("Numerical")
        # title("数值解")
        # grid(true)
        
        # subplot(1,3,3)
        # surf(X, Y, error)
        # xlabel("x"); ylabel("y"); zlabel("error")
        # title("误差")
        # grid(true)
    end

    # 计算收敛阶
    for k in 2:length(M)
        X = error_inf[k-1]/error_inf[k]
        Norm[k-1] = X 
    end

    # figure(length(N)+1)
    # plot(1:length(N)-1, Norm, "-b^")
    # xlabel("序号"); ylabel("误差阶数")
    # title("ADI格式误差阶")
    # grid(true)

    println("\n误差与收敛阶表：")
    @printf("%-10s %-10s %-15s %-10s\n", "m", "n", "error_inf", "收敛阶")
    for i in 1:length(M)
        if i == 1
            @printf("%-10d %-10d %-15.6e %-10s\n", M[i], N[i], error_inf[i], "-")
        else
            @printf("%-10d %-10d %-15.6e %-10.4f\n", M[i], N[i], error_inf[i], Norm[i-1])
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
# 调用函数
solve_adi_problem()