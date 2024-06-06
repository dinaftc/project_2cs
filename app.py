import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
import random
import subprocess
import ast
import astor
from pyparsing import Forward, Literal, Word, alphas, nums, alphanums
from deap import base, creator, tools, gp, algorithms
import operator
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProgramTestRequest(BaseModel):
    program: str
    test_cases: List[Tuple[List[int], int]]

population_size = 100
max_generations = 40

expr = Forward()
identifier = Word(alphas + '_', alphanums + '_')
operand = Word(nums) | (Literal('-').suppress() + Word(nums))
op = Literal('+') | Literal('-') | Literal('*') | Literal('/')
open_paren = Literal("(").suppress()
close_paren = Literal(")").suppress()
comma = Literal(",").suppress()
expr <<= op + open_paren + expr + comma + expr + close_paren | identifier | operand

def infix_action(tokens):
    if len(tokens) == 1:
        return tokens[0]
    elif tokens[0] in "+-*/":
        return f"({tokens[1]} {tokens[0]} {tokens[2]})"
    elif isinstance(tokens[1], str) and tokens[1].startswith("-"):
        return f"({tokens[1]}{tokens[0]}{tokens[2]})"
    else:
        return f"({tokens[0]}{tokens[1]})"

expr.setParseAction(infix_action)

def prefix_to_infix(prefix_expr):
    return expr.parseString(prefix_expr)[0]

def evaluate_program(erroneous_program, test_cases):
    with open('test_program.py', 'w') as f:
        f.write(erroneous_program)

    test_code = generate_unittest_class(erroneous_program, test_cases)
    with open('test_program_with_tests.py', 'w') as f:
        f.write(test_code)

    result = subprocess.run(['python', 'test_program_with_tests.py'], capture_output=True, text=True)
    output = result.stderr
    print(f"Evaluation Output: {output}")

    try:
        start_index = output.find('Ran ')
        end_index = output.find(' tests')
        if start_index == -1 or end_index == -1:
            raise ValueError("Invalid test output format")
        total_tests = int(output[start_index + 4:end_index].strip())
        failed_tests = output.count('FAIL')
        successful_tests = total_tests - failed_tests
        print(f"Total tests: {total_tests}, Failed tests: {failed_tests}, Successful tests: {successful_tests}")
    except Exception as e:
        print(f"Error parsing test results: {e}")
        total_tests = 25
        failed_tests = 25
        successful_tests = 0

    return total_tests, failed_tests, successful_tests

def replace_expression(code, line_number, old_expression, new_expression):
    tree = ast.parse(code)

    class ReplaceExpression(ast.NodeTransformer):
        def visit_Assign(self, node):
            if getattr(node, 'lineno', None) == line_number:
                try:
                    old_expr_ast = ast.parse("(" + old_expression + ")", mode='eval').body
                    new_expr_ast = ast.parse("(" + new_expression + ")", mode='eval').body
                except SyntaxError:
                    return node

                if self.compare_expr(node.value, old_expr_ast):
                    node.value = new_expr_ast

            return node

        def compare_expr(self, expr1, expr2):
            return ast.dump(expr1) == ast.dump(expr2)

    transformer = ReplaceExpression()
    transformed_tree = transformer.visit(tree)
    modified_code = astor.to_source(transformed_tree)

    return modified_code

def generate_unittest_class(wrong_program, test_cases):
    test_methods = ""
    for i, (input_vals, expected_output) in enumerate(test_cases):
        input_str = ", ".join(map(str, input_vals))
        test_methods += f"""
    def test_case_{i}(self):
        result = Gcd.gcd({input_str})
        self.assertEqual(result, {expected_output})
"""
    test_class_code = f"""
import unittest

class TestProgram(unittest.TestCase):
    {test_methods}

if __name__ == "__main__":
    unittest.main()
"""
    return f"{wrong_program}\n\n{test_class_code}"

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
pset = gp.PrimitiveSet("MAIN", arity=2)  # Set arity based on the number of variables
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def translate_expr(individual, variables):
    expr = str(individual)
    expr = expr.replace("add", "+")
    expr = expr.replace("sub", "-")
    expr = expr.replace("mul", "*")
    expr = expr.replace("rand101", str(random.randint(-10, 10)))
    for i, var in enumerate(variables):
        arg_name = f"ARG{i}"
        expr = expr.replace(arg_name, var)
    return expr

class TimeoutException(Exception):
    pass

def evalSymbReg(individual, variables, erroneous_program, test_cases):
    new_expression = translate_expr(individual, variables)
    new_expression = expr.parseString(str(new_expression))
    erroneous_code = replace_expression(erroneous_program, 11, "a - b", str(new_expression)[2:-2].strip())

    timeout_seconds = 1
    result = [25]  # Default to 25 failed tests in case of timeout

    def target():
        try:
            total_tests, failed_tests, successful_tests = evaluate_program(erroneous_code, test_cases)
            result[0] = failed_tests
        except Exception as e:
            print(f"Error evaluating program: {e}")
            result[0] = 25

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print("Timed out!")
        thread.join()  # Ensure the thread has finished

    return result[0],

def correct_program_with_genetic_programming(erroneous_program, test_cases):
    def extract_variables_constants(erroneous_program):
        tree = ast.parse(erroneous_program)
        variables = set()
        constants = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
                variables.add(node.id)
            elif isinstance(node, ast.Constant) and node.value != '__main__':
                if isinstance(node.value, (int, float, str)):
                    constants.add(node.value)
            elif isinstance(node, ast.Constant):
                constants.add(node.n)
        return list(variables), list(constants)

    variables, constants = extract_variables_constants(erroneous_program)
    if not variables:
        raise ValueError("No variables found in the program.")
    
    variables_list = list(variables)
    toolbox.register("evaluate", evalSymbReg, variables=variables_list, erroneous_program=erroneous_program, test_cases=test_cases)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, max_generations, stats=stats, halloffame=hof, verbose=True)

    best_individual = hof[0]
    best_expr_str = translate_expr(best_individual, variables_list)
    best_expr_str2 = expr.parseString(str(best_expr_str))
    best_expr_str2 = str(best_expr_str2)[2:-2].strip()

    return replace_expression(erroneous_program, 11, "a - b", best_expr_str2)

@app.post("/correct_program")
async def correct_program(request: ProgramTestRequest):
    erroneous_program = request.program
    test_cases = request.test_cases
    if not erroneous_program or not test_cases:
        raise HTTPException(status_code=400, detail="Invalid input")
    try:
        corrected_program = correct_program_with_genetic_programming(erroneous_program, test_cases)
        return {"corrected_program": corrected_program}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error correcting program: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)