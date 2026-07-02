// matmul.c — 作业 2：用 clang 生成矩阵乘法的 LLVM IR

#define N 4
void matmul(float A[N][N], float B[N][N], float C[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}

int main() {
    float A[N][N] = {{0}}, B[N][N] = {{0}}, C[N][N] = {{0}};
    matmul(A, B, C);
    return 0;
}
