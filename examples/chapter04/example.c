// example.c — 用 Clang 生成 LLVM IR 的示例

int add(int a, int b) { return a + b; }

int mul_add(int x, int y, int z) { return x * y + z; }

int factorial(int n) {
    if (n <= 1)
        return 1;
    return n * factorial(n - 1);
}

int main() {
    int arr[4] = {1, 2, 3, 4};
    int sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += arr[i];
    }
    return sum;
}
