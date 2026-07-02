"""简化版 code_generator visit_BinOp（第 3 章）"""
import ast


class CodeGen(ast.NodeVisitor):
    def visit_Name(self, node):
        return node.id

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return ("arith.addf", lhs, rhs)
        raise NotImplementedError(node.op)


tree = ast.parse("a + b")
gen = CodeGen()
assert gen.visit(tree.body[0].value) == ("arith.addf", "a", "b")
print("OK: codegen_visitor")
