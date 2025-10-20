function mjacobi(A, b, x, tol, max_it)
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
    
    for k = 1:max_it
        # Jacobi 迭代: x_new = D^{-1} * (b - (A - D) * x)
        x_new = D \ ((D - A) * x + b)
        
        # 更新 x
        x .= x_new
        
        # 计算残差和误差
        r = b - A * x
        err = norm(r) / bnrm2
        
        if err <= tol
            return x, k, err, time() - start_time
        end
    end
    
    return x, max_it, err, time() - start_time
end