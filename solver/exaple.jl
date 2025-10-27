using LinearAlgebra

include("mjacobi.jl")
include("mseidel.jl")
include("msor.jl")
include("mssor.jl")

# function main()
#     n = 2^12 - 1
#     alpha = ones(n) * 4
#     beta = ones(n-1) * (-1)
#     A = Tridiagonal(beta, alpha, beta)
#     b = [3.0; ones(n-2) * 2; 3.0]
#     x = zeros(n)
#     tol = 1e-10
#     max_it = 1000
#     ω = 1.1

#     k = zeros(4)
#     err = zeros(4)
#     elapsed_time = zeros(4)
#     x_result, k[1], err[1], elapsed_time[1] = mjacobi(A, b, x, tol, max_it)
#     x_result, k[2], err[2], elapsed_time[2] = mseidel(A, b, x, tol, max_it)
#     x_result, k[3], err[3], elapsed_time[3] = msor(A, b,ω , x, tol, max_it)
#     x_result, k[4], err[4], elapsed_time[4] = mssor(A, b,ω , x, tol, max_it)


#     figure(1)
#     plot(1:4,k)
#     figure(2)
#     plot(1:4,err)
#     figure(3)
#     plot(1:4,elapsed_time)

#     # 输出结果
#     println("\n迭代次数: ", k)
#     println("最终误差: ", err)
#     println("计算时间: ", elapsed_time, " 秒")

# end

# # 运行主程序
# main()

function main()
    n = 2^12 - 1
    alpha = ones(n) * 4
    beta = ones(n-1) * (-1)
    A = Tridiagonal(beta, alpha, beta)
    b = [3.0; ones(n-2) * 2; 3.0]
    tol = 1e-10
    max_it = 1000
    ω = 1.1

    methods = [
        ("Jacobi", (A, b, x, tol, max_it) -> mjacobi(A, b, x, tol, max_it)),
        ("Gauss-Seidel", (A, b, x, tol, max_it) -> mseidel(A, b, x, tol, max_it)),
        ("SOR", (A, b, x, tol, max_it) -> msor(A, b, ω, x, tol, max_it)),
        ("SSOR", (A, b, x, tol, max_it) -> mssor(A, b, ω, x, tol, max_it))
    ]

    results = []
    iterations = zeros(4)
    error = zeros(4)
    time = zeros(4)
    i = 1
    
    for (name, method) in methods
        x_initial = zeros(n)  # 确保每次都是新的初始向量
        x_result, iterations[i], error[i], time[i] = method(A, b, x_initial, tol, max_it)
        push!(results, (name, iterations[i], error[i], time[i]))
        println("$name: ")
        println("迭代次数: ", iterations[i])
        println("最终误差: ", error[i])
        println("计算时间: ", time[i], " 秒\n")

        i+=1
    end

    figure(1)
    plot(1:4,iterations)
    figure(2)
    plot(1:4,error)
    figure(3)
    plot(1:4,time)
end

main()