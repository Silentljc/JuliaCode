using LinearAlgebra

include("mjacobi.jl")
include("mseidel.jl")
include("msor.jl")

function main()
    n = 2^12 - 1
    alpha = ones(n) * 4
    beta = ones(n-1) * (-1)
    A = Tridiagonal(beta, alpha, beta)
    b = [3.0; ones(n-2) * 2; 3.0]
    x = zeros(n)
    tol = 1e-10
    max_it = 1000
    ω = 1.1

    # x_result, k, err, elapsed_time = mjacobi(A, b, x, tol, max_it)
    # x_result, k, err, elapsed_time = mseidel(A, b, x, tol, max_it)
    x_result, k, err, elapsed_time = msor(A, b,ω , x, tol, max_it)


    # 输出结果
    println("\n迭代次数: ", k)
    println("最终误差: ", err)
    println("计算时间: ", elapsed_time, " 秒")

end

# 运行主程序
main()
