function chase5(a, b, c, d, e, f)
    # 初始化
    n = length(c)
    s = zeros(n-2)
    m = zeros(n-1)
    l = zeros(n)
    p = zeros(n-1)
    q = zeros(n-2)
    
    x = zeros(n)
    y = zeros(n)
    
    # 对五对角矩阵LU分解
    m[1] = b[1]
    l[1] = c[1]
    l[2] = c[2] - b[1]*d[1]/c[1]
    p[1] = d[1]/c[1]
    q[1] = e[1]/l[1]
    q[2] = e[2]/l[2]
    p[2] = d[2]/l[2] - m[1]*q[1]/l[2]
    
    for i in 3:n
        s[i-2] = a[i-2]
        m[i-1] = b[i-1] - s[i-2]*p[i-2]
        l[i] = c[i] - s[i-2]*q[i-2] - m[i-1]*p[i-1]
        
        if i < n-1
            q[i] = e[i]/l[i]
            p[i] = (d[i] - m[i-1]*q[i-1])/l[i]
            continue
        end
        if i < n
            p[i] = (d[i] - m[i-1]*q[i-1])/l[i]
        end
    end
    
    # 追过程
    y[1] = f[1]/l[1]
    y[2] = (f[2] - m[1]*y[1])/l[2]
    for i in 3:n
        y[i] = (f[i] - s[i-2]*y[i-2] - m[i-1]*y[i-1])/l[i]
    end
    
    # 赶过程
    x[n] = y[n]
    x[n-1] = y[n-1] - p[n-1]*x[n]
    for i in n-2:-1:1
        x[i] = y[i] - p[i]*x[i+1] - q[i]*x[i+2]
    end
    
    return x
end