; sum_array.ll — 循环累加数组（含 main 供 lli 运行）

define i32 @sum_array(i32* %arr, i32 %len) {
entry:
    %result = alloca i32
    store i32 0, i32* %result
    %i = alloca i32
    store i32 0, i32* %i
    br label %loop_cond

loop_cond:
    %i_val = load i32, i32* %i
    %cond = icmp slt i32 %i_val, %len
    br i1 %cond, label %loop_body, label %loop_end

loop_body:
    %elem_ptr = getelementptr i32, i32* %arr, i32 %i_val
    %elem = load i32, i32* %elem_ptr
    %sum = load i32, i32* %result
    %new_sum = add i32 %sum, %elem
    store i32 %new_sum, i32* %result
    %next_i = add i32 %i_val, 1
    store i32 %next_i, i32* %i
    br label %loop_cond

loop_end:
    %final = load i32, i32* %result
    ret i32 %final
}

define i32 @main() {
    %arr = alloca [4 x i32]
    %p0 = getelementptr [4 x i32], [4 x i32]* %arr, i64 0, i64 0
    store i32 1, i32* %p0
    %p1 = getelementptr i32, i32* %p0, i64 1
    store i32 2, i32* %p1
    %p2 = getelementptr i32, i32* %p0, i64 2
    store i32 3, i32* %p2
    %p3 = getelementptr i32, i32* %p0, i64 3
    store i32 4, i32* %p3
    %sum = call i32 @sum_array(i32* %p0, i32 4)
    ret i32 %sum
}
