using LinearAlgebra

include("mjacobi.jl")
include("mseidel.jl")
include("msor.jl")
include("mssor.jl")

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