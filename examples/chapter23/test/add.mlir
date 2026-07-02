// RUN: mlir-opt %s --canonicalize | FileCheck %s
// CHECK: arith.addi

func.func @add(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  func.return %0 : i32
}
