from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Any
from fastapi.middleware.cors import CORSMiddleware
import re
import os
import ast
import astor
import textwrap
import asyncio
import logging
import traceback
import random
import operator
import builtins
import keyword
import csv
import time
import numpy as np
import unittest
from pyparsing import Forward, Word, alphas, alphanums, Literal, nums

from deap import algorithms, base, creator, tools, gp

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to specific origins you want to allow, e.g., ["http://localhost", "https://example.com"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# Helper functions and classes
def extract_class_and_method(program: str) -> Tuple[str, str]:
    class_name_search = re.search(r'class\s+(\w+)', program)
    method_name_search = re.search(r'def\s+(\w+)', program)

    if not class_name_search or not method_name_search:
        raise ValueError("Could not find class or method definition in the program.")

    class_name = class_name_search.group(1)
    method_name = method_name_search.group(1)

    return class_name, method_name

def generate_unittest_class(wrong_program: str, test_cases: List[Tuple[Tuple[Any, ...], Any]]) -> str:
    class_name, method_name = extract_class_and_method(wrong_program)
    test_class_name = "TestProgram"
    tests = []

    for i, test_case in enumerate(test_cases):
        inputs = ", ".join(map(str, test_case[0]))
        expected = test_case[1]
        test_code = f"""
    def test_case_{i}(self):
        result = {class_name}.{method_name}({inputs})
        self.assertEqual(result, {expected})
"""
        tests.append(test_code)

    test_class_code = f"""
import unittest

class {test_class_name}(unittest.TestCase):{''.join(tests)}

if __name__ == "__main__":
    unittest.main()
"""

    # Merge with the wrong program
    merged_code = f"{wrong_program}\n\n{test_class_code}"

    return merged_code

class ProgramTestRequest(BaseModel):
    program: str
    test_cases: List[Tuple[Tuple[Any, ...], Any]]
    line_number: int
    wrong_expression: str

# Constants
MAX_GENERATIONS = 50
POPULATION_SIZE = 100
TIMEOUT_SECONDS = 1  # Adjust the timeout as needed
OUTPUT_DIRECTORY = "."

# Define the primitive set for the symbolic regression problem
pset = gp.PrimitiveSet("MAIN", arity=1)  # This will be updated later based on variables
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addEphemeralConstant("rand101", lambda: random.randint(-10, 10))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

erroneous_program = ''

# Define the grammar for the prefix expression
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

async def evaluate_program(erroneous_program):
    try:
        proc = await asyncio.create_subprocess_exec(
            'python', '-c', erroneous_program,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            return 25, 25, 0  # Assuming all tests fail if it times out

        total_tests = 0
        total_failed_tests = 0

        stderr_output = stderr.decode()
        stdout_output = stdout.decode()

        match = re.search(r'Ran (\d+) tests', stderr_output)
        if match:
            total_tests = int(match.group(1))

        match_failures = re.search(r'FAILED \((failures=(\d+))?(, )?(errors=(\d+))?\)', stderr_output)
        if match_failures:
            failures = match_failures.group(2)
            errors = match_failures.group(5)
            if failures:
                total_failed_tests += int(failures)
            if errors:
                total_failed_tests += int(errors)

        successful_tests = total_tests - total_failed_tests
        failed_tests = total_failed_tests

        return total_tests, failed_tests, successful_tests

    except Exception as e:
        logging.error(f"Exception in evaluate_program: {e}")
        traceback.print_exc()
        return 25, 25, 0  # Default to all tests failing if there's an error

def replace_expression(code, line_number, old_expression, new_expression):
    lines = code.split('\n')
    if line_number > len(lines) or old_expression not in lines[line_number - 1]:
        logging.error("The specified line %d does not contain the old expression '%s'", line_number, old_expression)
        return code

    tree = ast.parse(code)

    class ReplaceExpression(ast.NodeTransformer):
        def visit_Assign(self, node):
            if getattr(node, 'lineno', None) == line_number:
                try:
                    old_expr_ast = ast.parse(f"({old_expression})", mode='eval').body
                    new_expr_ast = ast.parse(f"({new_expression})", mode='eval').body
                except SyntaxError as e:
                    logging.error("SyntaxError: %s", e)
                    return node

                if self.compare_expr(node.value, old_expr_ast):
                    logging.info("Replacing expression on line %d: %s -> %s", line_number, old_expression, new_expression)
                    node.value = new_expr_ast
            return node

        def compare_expr(self, expr1, expr2):
            return ast.dump(expr1) == ast.dump(expr2)

    transformer = ReplaceExpression()
    transformed_tree = transformer.visit(tree)

    modified_code = astor.to_source(transformed_tree)
    return modified_code

def extract_variables_constants(program):
    tree = ast.parse(program)
    variables = set()
    constants = set()

    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    module_level_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            module_level_names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_level_names.add(alias.name)
    builtins_names = set(dir(builtins))
    test_method_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            for child_node in ast.walk(node):
                if isinstance(child_node, ast.Name) and not isinstance(child_node.ctx, ast.Store):
                    test_method_names.add(child_node.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
            if node.id not in keyword.kwlist and node.id not in function_names \
               and node.id not in module_level_names and node.id not in builtins_names \
               and node.id not in test_method_names:
                variables.add(node.id)
        elif isinstance(node, ast.Constant) and node.value != '__main__':
            if isinstance(node.value, (int, float, str)):
                constants.add(node.value)
        elif isinstance(node, ast.Constant):
            constants.add(node.n)

    return list(variables), list(constants)

def translate_expr(individual, variables):
    expr = str(individual)
    for i, var in enumerate(variables):
        expr = expr.replace(f"ARG{i}", var)
    return expr

def evaluate_individual(individual, variables, program, line_number, wrong_expression):
    global erroneous_program

    infix_expression = translate_expr(individual, variables)
    new_program = replace_expression(program, line_number, wrong_expression, infix_expression)
    erroneous_program = new_program

    total_tests, failed_tests, successful_tests = asyncio.run(evaluate_program(new_program))

    return (failed_tests,)

toolbox.register("evaluate", evaluate_individual, variables=[], program='', line_number=0, wrong_expression='')
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

@app.post("/test_program")
async def test_program(request: ProgramTestRequest):
    global erroneous_program

    program = request.program
    test_cases = request.test_cases
    line_number = request.line_number
    wrong_expression = request.wrong_expression

    test_code = generate_unittest_class(program, test_cases)
    erroneous_program = test_code

    variables, constants = extract_variables_constants(program)
    pset = gp.PrimitiveSet("MAIN", arity=len(variables))

    for variable in variables:
        pset.addTerminal(variable)

    for constant in constants:
        pset.addTerminal(constant)

    toolbox.unregister("evaluate")
    toolbox.register("evaluate", evaluate_individual, variables=variables, program=program,
                     line_number=line_number, wrong_expression=wrong_expression)

    population = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    start_time = time.time()
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=MAX_GENERATIONS,
                                   halloffame=hof, verbose=True)
    end_time = time.time()
    elapsed_time = end_time - start_time

    best_individual = hof[0]
    best_expression = translate_expr(best_individual, variables)

    new_program = replace_expression(program, line_number, wrong_expression, best_expression)

    return JSONResponse(content={
        "original_program": program,
        "erroneous_program": erroneous_program,
        "new_program": new_program,
        "best_expression": best_expression,
        "elapsed_time": elapsed_time
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)