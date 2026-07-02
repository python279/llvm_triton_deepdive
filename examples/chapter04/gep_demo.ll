; gep_demo.ll — GEP 指令演示 + 作业 3 参考答案

%MyStruct = type { i32, [4 x float], i8* }

define void @gep_demo(%MyStruct* %arr) {
entry:
    ; 获取 arr[0] 的第一个字段（i32）
    %field0 = getelementptr %MyStruct, %MyStruct* %arr, i64 0, i32 0

    ; 获取 arr[0] 的第二个字段（[4 x float]）中第 2 个元素
    %elem = getelementptr %MyStruct, %MyStruct* %arr, i64 0, i32 1, i64 2

    ; 作业 3：访问 arr[3].field2[2]
    ; field2 = 结构体第 2 个字段（下标 1），再取数组下标 2
    %hw_answer = getelementptr %MyStruct, %MyStruct* %arr, i64 3, i32 1, i64 2

    ret void
}
