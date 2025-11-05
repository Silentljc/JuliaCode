using LinearAlgebra
include("chase.jl")
include("vp_chase.jl")

function main()
    start_time = time()
    a = fill(-2.0, 1023)   
    b = fill(4.0, 1024)    
    c = fill(-1.0, 1023)   
    f = [3.0; fill(1.0, 1022); 2.0]  
    l1 = 1
    u1 = 1
    
    # x = chase(a, b, c, f)
    x = vp_chase(a, b, c, f, l1, u1)
    
    # 构建三对角矩阵验证结果
    A = Tridiagonal(a, b, c)
    error = norm(f - A*x)

    elapsed_time = time() - start_time
    println("误差: ", error)
    println("程序共运行", elapsed_time, "秒")

end

main()