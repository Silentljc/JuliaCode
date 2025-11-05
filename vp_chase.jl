using LinearAlgebra

function vp_chase(a,b,c,f,l1,u1)

    # 选择合适的l1和u1，满足l1*u1 + 1 != 0即可

    n = length(b)
    d = zeros(n)
    l = zeros(n)
    u = zeros(n)
    g = zeros(n)
    t = zeros(n)

    l[1] = l1
    u[1] = u1
    d[1] = b[1]/(l[1]*u[1]+1)
    g[1] = f[1]/d[1]

    for k in 2:n
        u[k] = c[k-1]/d[k-1]
        d[k] = b[k]-a[k-1]*u[k]
        l[k] = a[k-1]/d[k]
        g[k] = f[k]/d[k]

    end

    s = zeros(n)
    s[n] = 1 + u[n]*l[n]
    t[n]  = g[n]

    for k in n-1:-1:1
        s[k] = 1 + u[k]*s[k+1]*l[k]
        t[k] = s[k+1]*g[k] - u[k+1]*t[k+1]

    end
    
    x = zeros(n)
    y = zeros(n+1)

    x[1] = t[1]/s[1]
    y[1] = u[1]*x[1]

    for k in 1:n
        y[k+1] = g[k] - l[k]*y[k]
    end

    x[n] = y[n+1]

    for k in n-1:-1:2
        x[k] = y[k+1] - u[k+1]*x[k+1]
    end
    return x

end
