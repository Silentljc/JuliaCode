function chase(a, b, c, d)
    # 追赶法求解三对角线性方程组 Ax = d
    # 输入参数：
    #   a - 下对角线元素向量 (长度 n-1)
    #   b - 主对角线元素向量 (长度 n)
    #   c - 上对角线元素向量 (长度 n-1)
    #   d - 右端项向量 (长度 n)
    # 输出：
    #   x - 解向量
    
    n = length(b)
    
    # 初始化
    alpha = zeros(n)
    beta  = zeros(n)
    x     = zeros(n)

    # 追过程
    alpha[1] = d[1] / b[1]
    beta[1]  = c[1] / b[1]
    for i in 2:(n - 1)
        denom = b[i] - a[i - 1] * beta[i - 1]
        alpha[i] = (d[i] - a[i - 1] * alpha[i - 1]) / denom
        beta[i]  = c[i] / denom
    end
    alpha[n] = (d[n] - a[n - 1] * alpha[n - 1]) / (b[n] - a[n - 1] * beta[n - 1])

    # 赶过程
    x[n] = alpha[n]
    for i in (n - 1):-1:1
        x[i] = alpha[i] - beta[i] * x[i + 1]
    end
    return x
end
