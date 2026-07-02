; factorial.ll — 递归计算阶乘（含 main 供 lli 运行）

define i32 @factorial(i32 %n) {
entry:
    %cond = icmp sle i32 %n, 1
    br i1 %cond, label %return, label %recurse

recurse:
    %sub = sub i32 %n, 1
    %sub_result = call i32 @factorial(i32 %sub)
    %result = mul i32 %n, %sub_result
    ret i32 %result

return:
    ret i32 1
}

define i32 @main() {
    %r = call i32 @factorial(i32 5)
    ret i32 %r
}
