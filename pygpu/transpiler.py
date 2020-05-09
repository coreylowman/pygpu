import inspect
import ast

RESULT = 'pygpu_result'
"""
the name of the result parameter for the cuda kernel
"""

INDEX = 'pygpu_index'
"""
the name of the index parameter for the cuda kernel
"""

"""
cuda symbol by python ast object
"""
SYMBOL_BY_OP_TYPE = {
    ast.Mult: '*',
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mod: '%',
    ast.Div: '/',
    ast.BitXor: '^',
    ast.BitAnd: '&',
    ast.BitOr: '|',
}


def transpile(fn):
    """
    Transpiles a python file object into a cuda kernel

    :param fn: A function pointer
    :return: source code string
    """
    source = inspect.getsource(fn)
    tree = ast.parse(source)
    ast_fn = tree.body[0]
    return CudaTranspiler().visit(ast_fn)


class CudaTranspiler(ast.NodeVisitor):
    """
    A python visitor that visits nodes in the python AST and transforms them into cuda code.
    """

    def __init__(self):
        # stores names of parameters so we can differentiate between parameters & local variables
        self.parameters = set()
        self.locals_variables = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Transpile a python function definition
        """
        # function definition
        lines = ['extern "C" __global__ void {}('.format(node.name)]
        for arg in node.args.args:
            # keep track of the parameter names so we know what to add indexes to later
            self.parameters.add(arg.arg)

            # add types for arguments
            if arg.annotation is None:
                lines.append('\tfloat *{},'.format(arg.arg))
            else:
                if 'id' in arg.annotation._fields:
                    lines.append('\t{} *{},'.format(arg.annotation.id, arg.arg))
                else:
                    raise ValueError('unknown annotation ' + arg.arg)

        # add in another parameter for the result
        lines.append('\tfloat *{}'.format(RESULT))

        # close function definition
        lines.append(') {')

        # generate index to use
        lines.append('\tconst unsigned int {} = (blockIdx.x * blockDim.x) + threadIdx.x;'.format(INDEX))

        # generate body lines, but don't add to lines yet - need to put in local variable declarations before body
        body_lines = []
        for statement in node.body:
            # note: first line in body is skipped because it is the whole function def? something not useful...
            body_lines.append('\t' + self.visit(statement))

        # initialize local variables to 0
        for v in self.locals_variables:
            lines.append('\tfloat {} = 0;'.format(v))
        lines.append('')

        # now append body lines
        lines.extend(body_lines)

        # finish function
        lines.append('}')

        return '\n'.join(lines)

    def visit_Assign(self, node: ast.Assign):
        # an assign... like `a = b`
        return '{} = {};'.format(self.visit(node.targets[0]), self.visit(node.value))

    def visit_AugAssign(self, node: ast.AugAssign):
        # an assign with an operator... like `a += b`
        return '{} {}= {};'.format(self.visit(node.target), SYMBOL_BY_OP_TYPE[type(node.op)], self.visit(node.value))

    def visit_BinOp(self, node: ast.BinOp):
        # a binary operator... like `a * b`
        return '({} {} {})'.format(self.visit(node.left), SYMBOL_BY_OP_TYPE[type(node.op)], self.visit(node.right))

    def visit_Expr(self, node: ast.Expr):
        # only docstring uses this so far... so return empty
        return ''

    def visit_Name(self, node: ast.Name):
        # a variable name usage... like `a`
        if node.id in self.parameters:
            return '{}[{}]'.format(node.id, INDEX)
        else:
            self.locals_variables.add(node.id)
            return node.id

    def visit_Num(self, node: ast.Num):
        # a raw number
        return '{}'.format(node.n)

    def visit_Return(self, node: ast.Return):
        # a return statement... replace this with setting the result parmeter
        return '{}[{}] = {};'.format(RESULT, INDEX, self.visit(node.value))

    def visit_Subscript(self, node: ast.Subscript):
        # subscript... like `a[i]`
        return '{}[{}]'.format(node.value.id, self.visit(node.slice.value))
