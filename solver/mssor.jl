function mssor(A, b, ω,x, tol, max_it)
    # 获取开始时间
    start_time = time()
    
    n = length(b)
    x_new = similar(x)
    
    # 计算初始残差
    r = b - A * x
    bnrm2 = norm(b)
    err = norm(r) / bnrm2
    
    if err < tol
        return x, 0, err, time() - start_time
    end
    
    # 提取对角矩阵
    D = Diagonal(diag(A))
    # 提取负严格下三角矩阵
    L = -tril(A,-1)
    # 提取负严格上三角矩阵
    U = -triu(A,1)
    
    for k = 1:max_it
        # SOR 迭代开始
        x_new = (D-ω*L) \ (((1-ω)*D+ω*U)*x+ω*b)
        
        # 更新 x
        x .=  (D-ω*U) \ (((1-ω)*D+ω*L)*x_new+ω*b)
        
        # 计算残差和误差
        r = b - A * x
        err = norm(r) / bnrm2
        
        if err <= tol
            return x, k, err, time() - start_time
        end
    end
    
    return x, max_it, err, time() - start_time
end