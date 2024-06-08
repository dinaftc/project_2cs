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
import threading
import queue

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

def replace_expression_condition(code, line_number, old_expression, new_expression):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logging.error(f"Syntax error while parsing code: {e}")
        return code

    class ReplaceExpression(ast.NodeTransformer):
        def visit_If(self, node):
            if getattr(node, 'lineno', None) == line_number:
                try:
                    old_expr_ast = ast.parse(old_expression, mode='eval').body
                    new_expr_ast = ast.parse(new_expression, mode='eval').body
                except SyntaxError as e:
                    logging.error(f"Syntax error in expressions: {e}")
                    return node

                if ast.dump(node.test) == ast.dump(old_expr_ast):
                    node.test = new_expr_ast

            self.generic_visit(node)
            return node

    transformer = ReplaceExpression()
    transformed_tree = transformer.visit(tree)
    modified_code = astor.to_source(transformed_tree)
    return modified_code

async def evaluate_program(erroneous_program):
    try:
        proc = await asyncio.create_subprocess_exec(
            'python', '-c', erroneous_program,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1)
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            return 25, 25, 0

        stderr_output = stderr.decode()
        stdout_output = stdout.decode()

        total_tests = 0
        total_failed_tests = 0

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
        return total_tests, total_failed_tests, successful_tests

    except Exception as e:
        logging.error(f"Exception in evaluate_program: {e}")
        traceback.print_exc()
        return 25, 25, 0

def prefix_to_infix(prefix_expr):
    return expr.parseString(prefix_expr)[0]

expr = Forward()
identifier = Word(alphas + '_', alphanums + '_')
operand = Word(nums) | (Literal('-').suppress() + Word(nums))
op = Literal('+') | Literal('-') | Literal('*') | Literal('/') | Literal('<=') | Literal('>=') | Literal('<') | Literal('==') | Literal('!=') | Literal('>') | Literal('and') | Literal('or')
open_paren = Literal("(").suppress()
close_paren = Literal(")").suppress()
comma = Literal(",").suppress()
expr <<= op + open_paren + expr + comma + expr + close_paren | identifier | operand

def infix_action(tokens):
    if len(tokens) == 1:
        return tokens[0]
    elif len(tokens) == 2:
        return f"{tokens[0]}({tokens[1]})"
    else:
        return f"({tokens[1]} {tokens[0]} {tokens[2]})"

expr.setParseAction(infix_action)

def extract_variables_constants(erroneous_program):
    try:
        tree = ast.parse(erroneous_program)
    except SyntaxError as e:
        logging.error(f"Syntax error while parsing code: {e}")
        return [], []

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
        elif isinstance(node, ast.Num):
            constants.add(node.n)

    return list(variables), list(constants)

class Bool(object):
    TRUE = True
    FALSE = False

class Int(int):
    def __bool__(self):
        return False

    def __new__(cls, value):
        if isinstance(value, bool):
            raise ValueError("Cannot create an Int instance from a bool")
        return super().__new__(cls, value)

    def __repr__(self):
        return f"Int({int(self)})"

pset = gp.PrimitiveSetTyped("MAIN", [Int, Int], Bool)
pset.addPrimitive(operator.lt, [Int, Int], Bool)
pset.addPrimitive(operator.le, [Int, Int], Bool)
pset.addPrimitive(operator.eq, [Int, Int], Bool)
pset.addPrimitive(operator.ne, [Int, Int], Bool)
pset.addPrimitive(operator.gt, [Int, Int], Bool)
pset.addPrimitive(operator.ge, [Int, Int], Bool)
pset.addPrimitive(operator.add, [Int, Int], Int)
pset.addPrimitive(operator.sub, [Int, Int], Int)
pset.addPrimitive(operator.mul, [Int, Int], Int)
pset.addPrimitive(operator.truediv, [Int, Int], Int)
pset.addPrimitive(operator.and_, [Bool, Bool], Bool)
pset.addPrimitive(operator.or_, [Bool, Bool], Bool)
pset.addTerminal(Bool.TRUE, Bool)
pset.addTerminal(Bool.FALSE, Bool)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate_program)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

@app.post("/api/program")
async def process_program(request: ProgramTestRequest):
    try:
        if not request.program.strip():
            raise HTTPException(status_code=400, detail="Program is empty.")
        if not request.test_cases:
            raise HTTPException(status_code=400, detail="Test cases are missing.")
        if not request.wrong_expression.strip():
            raise HTTPException(status_code=400, detail="Wrong expression is empty.")

        variables, constants = extract_variables_constants(request.program)
        if not variables:
            raise HTTPException(status_code=400, detail="No variables found in the program.")
        if not constants:
            raise HTTPException(status_code=400, detail="No constants found in the program.")

        for variable in variables:
            pset.addTerminal(variable, Int)

        for constant in constants:
            pset.addTerminal(constant, Int)

        def evaluate_individual(individual):
            new_expr = str(individual)
            infix_expr = prefix_to_infix(new_expr)
            modified_code = replace_expression_condition(request.program, request.line_number, request.wrong_expression, infix_expr)
            return evaluate_program(modified_code)

        population = toolbox.population(n=300)
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        for gen in range(NGEN):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = await asyncio.gather(*[evaluate_individual(ind) for ind in invalid_ind])
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

        best_individual = tools.selBest(population, 1)[0]
        best_solution = prefix_to_infix(str(best_individual))

        response_data = {"new_expression": best_solution}
        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logging.error(f"Exception in /api/program endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while processing the program.")
