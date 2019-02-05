from sympy import symbols
from sympy.vector import Vector
from itertools import product, combinations
import re


def main():
    s1, s2, a = symbols("s1 s2 a")
    vars = [s1, s2, a]
    order = 2
    terms = []
    for term_vars in product(*([(var, i) for i in range(order + 1)] for var in vars)):
        term = 1
        for var, power in term_vars:
            term *= var**power
        terms.append(str(term))
    generate_polynomial_cpp_func(vars, terms)


def generate_polynomial_cpp_func(vars, terms):
    func_params = ", ".join(f"double {var}" for var in vars)
    term_defs = []
    for term in terms:
        term_def = []
        for match in re.finditer(term_pattern, term):
            if match.group(2) is None:
                term_def.append(match.group(1))
            else:
                term_def.append("*".join([match.group(1)] * int(match.group(2))))
        term_def = "*".join(term_def)
        term_defs.append(term_def)

    term_defs = ",\n        ".join(term_defs)
    print(template.format(
        func_params=func_params,
        n_terms=len(terms),
        term_defs=term_defs
    ))


term_pattern = r"(\w+)(?:(?:\*\*)(\d+?))?"
template = """
const size_t n_terms = {n_terms};
typedef Eigen::Matrix<double, n_terms, 1> PolyRLVector;

PolyRLColVector PolyRLAgent::get_x({func_params}) {{
    PolyRLColVector x;
    x <<
        {term_defs}
    ;
    return x;
}}
"""

main()
