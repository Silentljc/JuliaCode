using LinearAlgebra
using Printf

# 追赶法求解三对角方程组
function chase(a, b, c, d)
    n = length(d)
    u = zeros(n)
    
    # 预处理
    c_prime = zeros(n-1)
    d_prime = zeros(n)
    
    c_prime[1] = c[1] / b[1]
    d_prime[1] = d[1] / b[1]
    
    for i in 2:n-1
        denominator = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denominator
    end
    
    d_prime[n] = (d[n] - a[n-1] * d_prime[n-1]) / (b[n] - a[n-1] * c_prime[n-1])
    
    # 回代
    u[n] = d_prime[n]
    for i in n-1:-1:1
        u[i] = d_prime[i] - c_prime[i] * u[i+1]
    end
    
    return u
end

function solve_adi_problem()
    start_time = time()
    
    # 测试不同的网格尺寸
    M_list = [10, 20, 40, 80, 160]
    error_inf = zeros(length(M_list))
    convergence_order = zeros(length(M_list)-1)
    
    for (idx, m) in enumerate(M_list)
        # 参数设置
        h = 1.0 / m
        τ = 1.0 / m  # 时间步长等于空间步长
        c = 1.0  # 波动方程中的常数
        
        # 网格定义
        x = collect(0:h:1)
        y = collect(0:h:1)
        t_final = 1.0
        t_steps = Int(round(t_final / τ))
        t = collect(0:τ:t_final)
        
        # 精确解函数
        exact_solution = (x, y, t) -> exp(0.5*(x+y) - t)
        
        # 初始条件函数
        u0 = (x, y) -> exp(0.5*(x+y))
        ψ = (x, y) -> -exp(0.5*(x+y))  # ∂u/∂t at t=0
        
        # 源项函数
        f = (x, y, t) -> 0.5 * exp(0.5*(x+y) - t)
        
        # ρ函数 (∂²u/∂t² at t=0)
        ρ = (x, y) -> begin
            # 由方程: u_tt = u_xx + u_yy + f
            # 在t=0时，u_xx = 0.25*exp(0.5*(x+y)), u_yy相同
            u_xx = 0.25 * exp(0.5*(x+y))
            u_yy = 0.25 * exp(0.5*(x+y))
            f_val = f(x, y, 0.0)
            return u_xx + u_yy + f_val
        end
        
        # 边界条件函数
        boundary_x0 = (y, t) -> exp(0.5*y - t)
        boundary_x1 = (y, t) -> exp(0.5*(1+y) - t)
        boundary_y0 = (x, t) -> exp(0.5*x - t)
        boundary_y1 = (x, t) -> exp(0.5*(1+x) - t)
        
        # 初始化数值解数组
        U = zeros(length(x), length(y), length(t))
        
        # 设置初始条件
        for i in 1:length(x)
            for j in 1:length(y)
                U[i, j, 1] = u0(x[i], y[j])
            end
        end
        
        # 设置边界条件
        for k in 1:length(t)
            for j in 1:length(y)
                U[1, j, k] = boundary_x0(y[j], t[k])
                U[end, j, k] = boundary_x1(y[j], t[k])
            end
            for i in 1:length(x)
                U[i, 1, k] = boundary_y0(x[i], t[k])
                U[i, end, k] = boundary_y1(x[i], t[k])
            end
        end
        
        # 计算系数
        r = (c^2 * τ^2) / (2 * h^2)
        
        # 构造三对角矩阵系数
        n_x = length(x) - 2  # 内部x网格点数
        n_y = length(y) - 2  # 内部y网格点数
        
        a_x = -r * ones(n_x - 1)
        b_x = (1 + 2r) * ones(n_x)
        c_x = -r * ones(n_x - 1)
        
        a_y = -r * ones(n_y - 1)
        b_y = (1 + 2r) * ones(n_y)
        c_y = -r * ones(n_y - 1)
        
        # 计算第一个时间层 (k=1) - 格式(5.68)
        # 第一步: 计算中间变量 ũ
        ũ = zeros(length(x), length(y))
        
        # 设置ũ的边界条件 - 格式(5.69)
        for j in 1:length(y)
            # ũ_{0j} = (I - (c²τ²)/2 δ_y²) u_{0j}^1
            if j == 1
                ũ[1, j] = U[1, j, 1] - r * (U[1, j, 1] - 2U[1, j, 1] + U[1, j+1, 1])
            elseif j == length(y)
                ũ[1, j] = U[1, j, 1] - r * (U[1, j-1, 1] - 2U[1, j, 1] + U[1, j, 1])
            else
                ũ[1, j] = U[1, j, 1] - r * (U[1, j-1, 1] - 2U[1, j, 1] + U[1, j+1, 1])
            end
            
            if j == 1
                ũ[end, j] = U[end, j, 1] - r * (U[end, j, 1] - 2U[end, j, 1] + U[end, j+1, 1])
            elseif j == length(y)
                ũ[end, j] = U[end, j, 1] - r * (U[end, j-1, 1] - 2U[end, j, 1] + U[end, j, 1])
            else
                ũ[end, j] = U[end, j, 1] - r * (U[end, j-1, 1] - 2U[end, j, 1] + U[end, j+1, 1])
            end
        end
        
        # 对每个固定的j，求解x方向 - 格式(5.70)
        for j in 2:length(y)-1
            rhs = zeros(n_x)
            for i in 2:length(x)-1
                # 右端项: u_{ij}^0 + τψ_{ij} - (τ³/3)ρ_{ij} + (τ²/2)f_{ij}^1
                rhs[i-1] = U[i, j, 1] + τ * ψ(x[i], y[j]) - (τ^3/3) * ρ(x[i], y[j]) + 
                          (τ^2/2) * f(x[i], y[j], t[2])
            end
            
            # 添加边界贡献
            rhs[1] -= (-r) * ũ[1, j]
            rhs[end] -= (-r) * ũ[end, j]
            
            # 求解三对角系统
            ũ[2:end-1, j] = chase(a_x, b_x, c_x, rhs)
        end
        
        # 第二步: 计算u^1 - 格式(5.71)-(5.72)
        for i in 2:length(x)-1
            rhs = zeros(n_y)
            for j in 2:length(y)-1
                rhs[j-1] = ũ[i, j]
            end
            
            # 边界条件 - 格式(5.71)
            rhs[1] -= (-r) * U[i, 1, 2]  # u_{i0}^1 = α(x_i, y_0, t_1)
            rhs[end] -= (-r) * U[i, end, 2]  # u_{i,m2}^1 = α(x_i, y_m2, t_1)
            
            # 求解三对角系统
            U[i, 2:end-1, 2] = chase(a_y, b_y, c_y, rhs)
        end
        
        # 计算后续时间层 (k ≥ 2) - 格式(5.73)
        for k in 2:length(t)-1
            # 第一步: 计算中间变量 ū
            ū = zeros(length(x), length(y))
            
            # 设置ū的边界条件 - 格式(5.74)
            for j in 1:length(y)
                # ū_{0j} = (I - (c²τ²)/2 δ_y²) u_{0j}^k
                if j == 1
                    ū[1, j] = U[1, j, k] - r * (U[1, j, k] - 2U[1, j, k] + U[1, j+1, k])
                elseif j == length(y)
                    ū[1, j] = U[1, j, k] - r * (U[1, j-1, k] - 2U[1, j, k] + U[1, j, k])
                else
                    ū[1, j] = U[1, j, k] - r * (U[1, j-1, k] - 2U[1, j, k] + U[1, j+1, k])
                end
                
                if j == 1
                    ū[end, j] = U[end, j, k] - r * (U[end, j, k] - 2U[end, j, k] + U[end, j+1, k])
                elseif j == length(y)
                    ū[end, j] = U[end, j, k] - r * (U[end, j-1, k] - 2U[end, j, k] + U[end, j, k])
                else
                    ū[end, j] = U[end, j, k] - r * (U[end, j-1, k] - 2U[end, j, k] + U[end, j+1, k])
                end
            end
            
            # 对每个固定的j，求解x方向 - 格式(5.75)
            for j in 2:length(y)-1
                rhs = zeros(n_x)
                for i in 2:length(x)-1
                    # 右端项: 2u_{ij}^k - u_{ij}^{k-1} + (τ²/2)f_{ij}^k
                    rhs[i-1] = 2U[i, j, k] - U[i, j, k-1] + (τ^2/2) * f(x[i], y[j], t[k])
                end
                
                # 添加边界贡献
                rhs[1] -= (-r) * ū[1, j]
                rhs[end] -= (-r) * ū[end, j]
                
                # 求解三对角系统
                ū[2:end-1, j] = chase(a_x, b_x, c_x, rhs)
            end
            
            # 第二步: 计算u^{k+1} - 格式(5.76)-(5.77)
            for i in 2:length(x)-1
                rhs = zeros(n_y)
                for j in 2:length(y)-1
                    rhs[j-1] = ū[i, j]
                end
                
                # 边界条件 - 格式(5.76)
                bc_y0 = 0.5 * (boundary_y0(x[i], t[k+1]) + boundary_y0(x[i], t[k-1]))
                bc_y1 = 0.5 * (boundary_y1(x[i], t[k+1]) + boundary_y1(x[i], t[k-1]))
                
                rhs[1] -= (-r) * bc_y0
                rhs[end] -= (-r) * bc_y1
                
                # 求解三对角系统
                U[i, 2:end-1, k+1] = chase(a_y, b_y, c_y, rhs)
            end
        end
        
        # 计算最大误差
        max_error = 0.0
        for i in 1:length(x)
            for j in 1:length(y)
                for k in 1:length(t)
                    exact_val = exact_solution(x[i], y[j], t[k])
                    error_val = abs(U[i, j, k] - exact_val)
                    if error_val > max_error
                        max_error = error_val
                    end
                end
            end
        end
        error_inf[idx] = max_error
        
        println("m = $m, 最大误差 = $(error_inf[idx])")
    end
    
    # 计算收敛阶
    for i in 2:length(M_list)
        convergence_order[i-1] = log2(error_inf[i-1] / error_inf[i])
    end
    
    # 输出结果
    println("\n收敛性分析:")
    println("m\t\t最大误差\t\t收敛阶")
    for i in 1:length(M_list)
        if i == 1
            @printf("%d\t\t%.2e\t\t-\n", M_list[i], error_inf[i])
        else
            @printf("%d\t\t%.2e\t\t%.4f\n", M_list[i], error_inf[i], convergence_order[i-1])
        end
    end
    
    elapsed_time = time() - start_time
    println("\n计算完成，耗时: $(round(elapsed_time, digits=2)) 秒")
    
    return error_inf, convergence_order
end

# 运行求解
error_inf, convergence_order = solve_adi_problem()