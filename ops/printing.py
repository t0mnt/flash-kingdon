import sympy
from sympy.printing.pycode import PythonCodePrinter, ArrayPrinter
import triton.language
import math

class TritonPrinter(ArrayPrinter, PythonCodePrinter):
    _module = 'triton.language'
    namespace = {**triton.language.__dict__}

    def _print_Pow(self, expr, rational=False):
        if expr.exp == 2:
            return f'{expr.base}*{expr.base}'
        if expr.exp == 0.5:
            return self._print_sqrt(expr.base)
        return super()._print_Pow(expr, rational=rational)

    def _print_erf(self, expr):
        # print(self.namespace['erf'])
        return f'erf({self._print(expr.args[0])})'

    def _print_exp(self, expr):
        return f'exp({self._print(expr.args[0])})'

    def _print_sqrt(self, expr):
        return f'sqrt{self._print(expr.args[0])}'
    
    def _print_Pi(self, expr):
        return f'{math.pi}'

