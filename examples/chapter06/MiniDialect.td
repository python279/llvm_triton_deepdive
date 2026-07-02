#ifndef MINI_DIALECT_TD
#define MINI_DIALECT_TD

include "mlir/IR/OpBase.td"

def Mini_Dialect : Dialect {
  let name = "mini";
  let cppNamespace = "::mini";
}

class Mini_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mini_Dialect, mnemonic, traits>;

def Mini_AddOp : Mini_Op<"add"> {
  let summary = "Elementwise add";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Mini_MulOp : Mini_Op<"mul"> {
  let summary = "Elementwise mul";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

#endif
