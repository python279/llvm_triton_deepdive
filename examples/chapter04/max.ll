; max.ll — 作业 1：返回两个 i32 中的较大值

define i32 @max(i32 %a, i32 %b) {
entry:
    %cond = icmp sgt i32 %a, %b
    %r = select i1 %cond, i32 %a, i32 %b
    ret i32 %r
}

define i32 @main() {
    %r = call i32 @max(i32 3, i32 7)
    ret i32 %r
}
