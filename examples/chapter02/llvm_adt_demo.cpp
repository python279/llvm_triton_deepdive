#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

int main() {
    llvm::SmallVector<int, 4> v = {1, 2, 3};
    int sum = 0;
    for (int x : v)
        sum += x;
    llvm::errs() << "sum=" << sum << "\n";
    return sum == 6 ? 0 : 1;
}
