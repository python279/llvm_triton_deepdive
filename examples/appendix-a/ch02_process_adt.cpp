// 第 2 章作业 1：用 LLVM ADT 重写 process()
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>

using namespace llvm;

SmallVector<int> process(ArrayRef<int> values) {
    SmallVector<int> result;
    for (auto [i, v] : llvm::enumerate(values)) {
        (void)i;
        result.push_back(v * 2);
    }
    return result;
}

int main() {
    int arr[] = {1, 2, 3};
    auto r = process(arr);
    for (int x : r)
        std::cout << x;
    std::cout << '\n';
    return (r.size() == 3 && r[0] == 2 && r[1] == 4 && r[2] == 6) ? 0 : 1;
}
