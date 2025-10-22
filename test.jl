function loop_test()
    println("=== Julia 循环性能测试 ===")
    
    n = 10_000_000  # 1000万次迭代
    
    # 测试1: 简单累加
    time1 = @elapsed begin
        sum_val = 0.0
        for i in 1:n
            sum_val += i
        end
    end
    println("简单累加: $(round(time1, digits=4)) 秒")
    
    # 测试2: 数学函数计算
    time2 = @elapsed begin
        sum_val = 0.0
        for i in 1:n
            sum_val += sin(i) * cos(i)
        end
    end
    println("三角函数计算: $(round(time2, digits=4)) 秒")
end

# 运行测试
loop_test()