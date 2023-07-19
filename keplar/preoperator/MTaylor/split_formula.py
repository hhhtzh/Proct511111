import re
from sympy import symbols, Eq, sympify
"""
ZJW did the following 2023/07/19. 
The function separates a formula into different formulas based on different subsets.
"""
def split_formula(formula):
    # Remove spaces from the formula
    formula = formula.replace(" ", "")
    formula = "+" + formula.strip()

    terms = []
    term = ""
    operator = "+"

    # Split the formula into individual terms
    for i in range(len(formula)):
        char = formula[i]
        if char == "+" or char == "-":
            if term != "":
                terms.append(operator + term)
            term = ""
            operator = char
        else:
            term += char

    if term != "":
        terms.append(operator + term)

    return terms


def assign_terms_to_equations(terms, subsets):
    equations = {}
    equations[0] = [] # 用于存储不属于任何子集的项

    # Assign terms to equations based on subsets
    for term in terms:
        subsets_count = 0
        subset_index = -1
        is_exclusive = False
        is_inclusive = True

        # Count the number of subsets the term belongs to
        for i, subset in enumerate(subsets):
            if any(element in term for element in subset):
                subsets_count += 1
                subset_index = i
                is_inclusive = False

        # Check if the term belongs exclusively to one subset
        if subsets_count == 1:
            is_exclusive = True

        # Check if the term belongs to any other subset
        for i, subset in enumerate(subsets):
            if i != subset_index:
                if any(element in term for element in subset):
                    is_exclusive = False
                    break

        # Add the term to the corresponding equation
        if is_exclusive:
            if subset_index not in equations:
                equations[subset_index] = []
            
            equations[subset_index].append(term)

        if is_inclusive:
            equations[0].append(term)

    formula = ""

    # Combine the terms of each equation
    for subset_index, equation_terms in equations.items():
        equation = "".join(equation_terms)
        formula += equation + "\n"

    return formula.strip()


formula = "3*x0 - x1 + 2*x2*x3 + x2 - 4*x3 + 2*x4 + 9"
subset1 = ['x0', 'x2']
subset2 = ['x1', 'x3']
subset3 = ['x4']
subsets = [subset1, subset2, subset3]

# Split the formula into individual terms
terms = split_formula(formula)
print(f"Terms: {terms}")

# Assign terms to equations based on subsets
equations = assign_terms_to_equations(terms, subsets)
print(f"Equations: {equations}")

# Get the first equation from the equations string
equation_str = equations.split('\n')[0]

# Create variables using subset1
vars = symbols(' '.join(subset1))

# Convert the equation string to a sympy expression
equation = Eq(sympify(equation_str), 0)

# Print the equation
print(equation)
