; 第 3 章作业 1：SSA 形式（已验证：llvm-as 通过）

define i32 @example(i32 %x, i32 %y) {
entry:
    %z1 = add i32 %x, %y
    %cond = icmp sgt i32 %z1, 0
    br i1 %cond, label %then, label %else

then:
    %z2 = mul i32 %z1, 2
    br label %merge

else:
    %z3 = mul i32 %z1, 3
    br label %merge

merge:
    %z = phi i32 [%z2, %then], [%z3, %else]
    ret i32 %z
}
