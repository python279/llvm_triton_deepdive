define i32 @example(i32 %x, i32 %y) {
entry:
    %z0 = add i32 %x, %y
    %cond = icmp sgt i32 %z0, 0
    br i1 %cond, label %then, label %else

then:
    %z1 = mul i32 %z0, 2
    br label %merge

else:
    %z2 = mul i32 %z0, 3
    br label %merge

merge:
    %z = phi i32 [ %z1, %then ], [ %z2, %else ]
    ret i32 %z
}

define i32 @main() {
    %r = call i32 @example(i32 3, i32 4)
    ret i32 %r
}
