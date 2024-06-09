import random
import operator
import builtins
import keyword
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
import queue
import subprocess
import os
import ast
import astor
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

class CorrectionRequest(BaseModel):
    program: str
    wrong_expression: str
    line_number: int
    test_cases: str

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

# Define the primitive set for the symbolic regression problem
pset = gp.PrimitiveSetTyped("MAIN", [Int, Int], Bool)  # Set arity equal to the number of variables

# Add comparison operators which take two integers and return a boolean
pset.addPrimitive(operator.lt, [Int, Int], Bool)
pset.addPrimitive(operator.le, [Int, Int], Bool)
pset.addPrimitive(operator.eq, [Int, Int], Bool)
pset.addPrimitive(operator.ne, [Int, Int], Bool)
pset.addPrimitive(operator.gt, [Int, Int], Bool)
pset.addPrimitive(operator.ge, [Int, Int], Bool)
pset.addPrimitive(operator.add, [Int, Int], Int)  # Addition (+)
pset.addPrimitive(operator.sub, [Int, Int], Int)  # Subtraction (-)
pset.addPrimitive(operator.mul, [Int, Int], Int)

# Add logical operators which take two booleans and return a boolean
pset.addPrimitive(operator.and_, [Bool, Bool], Bool)
pset.addPrimitive(operator.or_, [Bool, Bool], Bool)
# Add terminals: True and False of type bool, and ephemeral constants of type int
pset.addTerminal(True, Bool)
pset.addTerminal(False, Bool)
pset.addEphemeralConstant("rand101", lambda: random.randint(0, 1), Int)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

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
    elif len(tokens) == 2:  # Unary operators like '-' or 'not'
        return f"{tokens[0]}({tokens[1]})"
    else:
        return f"({tokens[1]} {tokens[0]} {tokens[2]})"

expr.setParseAction(infix_action)

def translate_expr(individual, variables):
    expr = str(individual)
    expr = expr.replace("eq", "==") 
    expr = expr.replace("add", "+")
    expr = expr.replace("sub", "-")
    expr = expr.replace("mul", "*")
    expr = expr.replace("lt", "<")
    expr = expr.replace("le", "<=")
    expr = expr.replace("ne", "!=")
    expr = expr.replace("gt", ">")
    expr = expr.replace("ge", ">=")
    expr = expr.replace("and_", "and")
    expr = expr.replace("or_", "or")
    expr = expr.replace("rand101", str(random.randint(-10, 10)))
    for i, var in enumerate(variables):
        arg_name = "ARG{}".format(i)
        expr = expr.replace(arg_name, var)
    return expr

def get_variable_name(node, variables):
    if isinstance(node, ast.Name):
        if node.id in variables:
            return node.id
    return None

def has_duplicate_vars(expr, variables):
    try:
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp)):
                if isinstance(node, ast.BinOp):
                    left = get_variable_name(node.left, variables)
                    right = get_variable_name(node.right, variables)
                    if left == right:
                        return True
                elif isinstance(node, ast.Compare):
                    left = get_variable_name(node.left, variables)
                    for comparator in node.comparators:
                        right = get_variable_name(comparator, variables)
                        if left == right:
                            return True
    except (SyntaxError, ValueError):
        pass
    return False

def eval_individual(individual, variables, result_queue, timeout_event, erroneous_program, line_number, old_expression):
    try:
        new_expression = translate_expr(individual, variables)
        new_expression = expr.parseString(str(new_expression))
        print("Before Replacement:")
        print(erroneous_program)
        erroneous_program = replace_expression_condition(erroneous_program, line_number, old_expression, str(new_expression)[2:-2].strip())
        print("After Replacement:")
        print(erroneous_program)
        total_tests, failed_tests, successful_tests = evaluate_program(erroneous_program)
        print(f"Failed Tests: {failed_tests}, Total Tests: {total_tests}, Successful Tests: {successful_tests}")
        if has_duplicate_vars(str(new_expression)[2:-2].strip(), variables):
            failed_tests += 10
        result_queue.put((failed_tests, total_tests))
    except Exception as e:
        print(f"Evaluation error: {e}")
        result_queue.put((float('inf'), 0))
    finally:
        timeout_event.set()

def evalSymbReg(individual, variables, erroneous_program, line_number, old_expression, timeout=3):
    timeout_event = threading.Event()
    result_queue = queue.Queue()
    eval_thread = threading.Thread(target=eval_individual, args=(individual, variables, result_queue, timeout_event, erroneous_program, line_number, old_expression))
    eval_thread.start()
    eval_thread.join(timeout=timeout)
    if not timeout_event.is_set():
        print("Evaluation timed out.")
        return float('inf'), 0
    evaluation_result = result_queue.get()
    if evaluation_result is None:
        print("Evaluation failed.")
        return float('inf'), 0
    failed_tests, total_tests = evaluation_result
    return failed_tests,

def extract_variables_constants(erroneous_program):
    tree = ast.parse(erroneous_program)
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
            if node.id not in keyword.kwlist and node.id not in function_names and node.id not in module_level_names and node.id not in builtins_names and node.id not in test_method_names:
                variables.add(node.id)
        elif isinstance(node, ast.Constant) and node.value != '__main__':
            if isinstance(node.value, (int, float, str)):
                constants.add(node.value)
        elif isinstance(node, ast.Num):
            constants.add(node.n)
    return list(variables), list(constants)

def replace_expression_condition(code, line_number, old_expression, new_expression):
    tree = ast.parse(code)
    class ReplaceExpression(ast.NodeTransformer):
        def visit_If(self, node):
            if getattr(node, 'lineno', None) == line_number:
                try:
                    old_expr_ast = ast.parse(old_expression, mode='eval').body
                    new_expr_ast = ast.parse(new_expression, mode='eval').body
                except SyntaxError as e:
                    print("SyntaxError:", e)
                    return node
                if self.compare_expr(node.test, old_expr_ast):
                    node.test = new_expr_ast
            self.generic_visit(node)
            return node
        def compare_expr(self, expr1, expr2):
            return ast.dump(expr1) == ast.dump(expr2)
    transformer = ReplaceExpression()
    transformed_tree = transformer.visit(tree)
    modified_code = astor.to_source(transformed_tree)
    return modified_code

def evaluate_program(erroneous_program):
    # Execute the erroneous program as a subprocess
    result = subprocess.run(['python', '-c', erroneous_program], capture_output=True, text=True)

    # Initialize counters
    total_tests = 0
    total_failed_tests = 0

    # Extract information from the stderr
    stderr_output = result.stderr

    # Use regex to find the number of tests ran
    match = re.search(r'Ran (\d+) tests', stderr_output)
    if match:
        total_tests = int(match.group(1))

    # Use regex to find the number of failures and errors
    match_failures = re.search(r'FAILED \((failures=(\d+))?(, )?(errors=(\d+))?\)', stderr_output)
    if match_failures:
        failures = match_failures.group(2)
        errors = match_failures.group(5)
        if failures:
            total_failed_tests += int(failures)
        if errors:
            total_failed_tests += int(errors)

    # Calculate the number of successful tests
    successful_tests = total_tests - total_failed_tests
    failed_tests = total_failed_tests

    print('total_tests : ', total_tests)
    print('failed_tests (including errors) : ', total_failed_tests)
    print('successful_tests : ', successful_tests)

    return total_tests, failed_tests, successful_tests

@app.post("/correct_program")
def correct_program(request: CorrectionRequest):
    erroneous_program = request.program
    line_number = request.line_number
    wrong_expression = request.wrong_expression
    test_cases = request.test_cases
    variables, _ = extract_variables_constants(erroneous_program)
    if len(variables) < 1:
        raise HTTPException(status_code=400, detail="Incorrect number of variables extracted.")
    variables_list = list(variables)
    toolbox.register("evaluate", evalSymbReg, variables=variables_list, erroneous_program=erroneous_program, line_number=line_number, old_expression=wrong_expression)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    ngen = 10
    best_gen = -1
    for gen in range(ngen):
        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 1, stats=stats, halloffame=hof, verbose=True)
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]
        failed_tests = toolbox.evaluate(best_individual, variables=variables_list, erroneous_program=erroneous_program, line_number=line_number)
        if best_fitness == 0:
            best_gen = gen
            break
    if best_gen != -1:
        best_individual = hof[0]
        best_expr_str = translate_expr(best_individual, variables_list)
        best_expr_str2 = expr.parseString(str(best_expr_str))
        return {"best_generation": best_gen, "corrected_expression": str(best_expr_str2)}
    else:
        raise HTTPException(status_code=500, detail="No solution found within the time limit.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
