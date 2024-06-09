from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Any
from fastapi.middleware.cors import CORSMiddleware
import re
import os
import subprocess
import ast
import astor
import textwrap
import asyncio
import logging
import traceback
import random
import threading
import queue
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



# Constants
MAX_GENERATIONS = 50
POPULATION_SIZE = 200
TIMEOUT_SECONDS = 1  # Adjust the timeout as needed
OUTPUT_DIRECTORY = "."

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
pset.addPrimitive(operator.add, [Int, Int],Int)  # Addition (+)
pset.addPrimitive(operator.sub, [Int, Int],Int)  # Subtraction (-)
pset.addPrimitive(operator.mul, [Int, Int],Int)

# Add logical operators which take two booleans and return a boolean
pset.addPrimitive(operator.and_, [Bool, Bool], Bool)
pset.addPrimitive(operator.or_, [Bool, Bool], Bool)
# Add terminals: True and False of type bool, and ephemeral constants of type int
pset.addTerminal(True,Bool)
pset.addTerminal(False,Bool)
pset.addEphemeralConstant("rand101", lambda: random.randint(0, 1), Int)

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
op = Literal('+') | Literal('-') | Literal('*') | Literal('/') | Literal('<=') | Literal('>=') | Literal('<') | Literal('==') | Literal('!=') | Literal('>') | Literal('and') | Literal('or')
open_paren = Literal("(").suppress()
close_paren = Literal(")").suppress()
comma = Literal(",").suppress()
expr <<= op + open_paren + expr + comma + expr + close_paren | identifier | operand

# Define the infix notation with the correct operator precedence
def infix_action(tokens):
    if len(tokens) == 1:
        return tokens[0]
    elif len(tokens) == 2:  # Unary operators like '-' or 'not'
        return f"{tokens[0]}({tokens[1]})"
    else:
        return f"({tokens[1]} {tokens[0]} {tokens[2]})"

expr.setParseAction(infix_action)

def prefix_to_infix(prefix_expr):
    return expr.parseString(prefix_expr)[0]

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


def replace_expression_condition(code, line_number, old_expression, new_expression):
    # Parse the code into an abstract syntax tree (AST)
    print("Old Expression:", old_expression)
    print("New Expression:", new_expression)
    tree = ast.parse(code)

    # Define a visitor to traverse the AST and perform replacements
    class ReplaceExpression(ast.NodeTransformer):
        def visit_If(self, node):
            # Check if the node corresponds to the specified line number
            if getattr(node, 'lineno', None) == line_number:
                # Try parsing the old and new expressions into AST nodes
                try:
                    old_expr_ast = ast.parse(old_expression, mode='eval').body
                    new_expr_ast = ast.parse(new_expression, mode='eval').body
                except SyntaxError as e:
                    print("SyntaxError:", e)
                    return node

                # Compare the old expression in the AST with the given old expression
                if self.compare_expr(node.test, old_expr_ast):
                    # Replace the old expression with the new expression
                    node.test = new_expr_ast

            # Continue to traverse the child nodes
            self.generic_visit(node)
            return node

        def compare_expr(self, expr1, expr2):
            # Compare the string representations of the AST nodes
            return ast.dump(expr1) == ast.dump(expr2)

    # Instantiate the visitor and traverse the AST
    transformer = ReplaceExpression()
    transformed_tree = transformer.visit(tree)

    # Generate Python code from the modified AST
    modified_code = astor.to_source(transformed_tree)
    print(modified_code)
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
    # Replace the primitive set names with mathematical operators
    expr = expr.replace("eq", "==") # Equal to
    expr = expr.replace("add", "+")
    expr = expr.replace("sub", "-")
    expr = expr.replace("mul", "*")
    expr = expr.replace("lt", "<")  # Less than
    expr = expr.replace("le", "<=") # Less than or equal to
    expr = expr.replace("ne", "!=") # Not equal to
    expr = expr.replace("gt", ">")  # Greater than
    expr = expr.replace("ge", ">=")
    expr = expr.replace("and_", "and")  # Logical AND
    expr = expr.replace("or_", "or")
    # Replace ephemeral constants with their values
    expr = expr.replace("rand101", str(random.randint(-10, 10)))
    # Replace ARG0, ARG1, ..., ARGn with the variable names
    for i, var in enumerate(variables):
        arg_name = "ARG{}".format(i)
        # Replace the argument name with the variable name
        expr = expr.replace(arg_name, var)
    return expr

def get_variable_name(node, variables):
    """
    Helper function to get the variable name from a node, if it exists in the given variables list.
    """
    if isinstance(node, ast.Name):
        if node.id in variables:
            return node.id
    return None

def has_duplicate_vars(expr, variables):
    """
    Check if an expression contains duplicate variables in a direct comparison.
    """
    try:
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                left = get_variable_name(node.left, variables)
                for comparator in node.comparators:
                    right = get_variable_name(comparator, variables)
                    if left and left == right:
                        return True
    except (SyntaxError, ValueError):
        pass
    return False

def eval_individual(individual, variables ,line_number ,wrong_expression, result_queue, timeout_event):
    try:
        new_expression = translate_expr(individual, variables)
        new_expression = expr.parseString(str(new_expression))
        erroneous_code = replace_expression_condition(erroneous_program, line_number, wrong_expression, str(new_expression)[2:-2].strip())
        # print(erroneous_code)
        total_tests, failed_tests, successful_tests = evaluate_program(erroneous_code)
        if has_duplicate_vars(str(new_expression)[2:-2].strip(), variables):
          # Apply a penalty to the fitness score
          failed_tests += 10
        result_queue.put((failed_tests, total_tests))
    except Exception as e:
        print(f"Evaluation error: {e}")
        result_queue.put((float('inf'), 0))
    finally:
        timeout_event.set()  # Set the event to signal the end of evaluation

def evalSymbReg(individual, variables,line_number,wrong_expression, timeout=3):
    timeout_event = threading.Event()  # Event to signal timeout
    result_queue = queue.Queue()  # Queue to store evaluation result

    # Start a new thread to execute eval_individual
    eval_thread = threading.Thread(target=eval_individual, args=(individual, variables,line_number,wrong_expression, result_queue, timeout_event))
    eval_thread.start()

    # Wait for the thread to finish or timeout
    eval_thread.join(timeout=timeout)  # Timeout set to the given seconds

    if not timeout_event.is_set():
        print("Evaluation timed out.")
        return float('inf'), 0  # Return infinity and 0 if evaluation timed out

    evaluation_result = result_queue.get()  # Get the result from the queue
    if evaluation_result is None:
        print("Evaluation failed.")
        return float('inf'), 0  # Return infinity and 0 if evaluation failed

    failed_tests, total_tests = evaluation_result
    return failed_tests,

def evolution(toolbox, pop, hof, stats,variables, ngen):
    best_gen = -1
    for gen in range(ngen):
        # Apply the evolutionary algorithm
        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 1, stats=stats, halloffame=hof, verbose=True)

        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]

        # Evaluate the best individual
        failed_tests = toolbox.evaluate(best_individual, variables=variables)

        if best_fitness == 0:  # Fitness 0 means all tests passed
            best_gen = gen
            print(f"All test cases passed at generation {gen}")
            break

    return best_gen

def get_expression_at_line(code, line_number):
    # Parse the code into an abstract syntax tree (AST)
    tree = ast.parse(code)

    # Define a visitor to traverse the AST and find the expression
    class FindExpression(ast.NodeVisitor):
        def __init__(self):
            self.expression = None

        def visit_Assign(self, node):
            # Check if the node corresponds to the specified line number
            if getattr(node, 'lineno', None) == line_number:
                self.expression = node.value
            self.generic_visit(node)

        def visit_If(self, node):
            if getattr(node, 'lineno', None) == line_number:
                self.expression = node.test
            self.generic_visit(node)

        def visit_While(self, node):
            if getattr(node, 'lineno', None) == line_number:
                self.expression = node.test
            self.generic_visit(node)

    # Instantiate the visitor and traverse the AST
    finder = FindExpression()
    finder.visit(tree)

    # Convert the found expression back to source code
    if finder.expression is not None:
        expression_code = astor.to_source(finder.expression).strip()
        return expression_code
    else:
        return None

@app.post("/test_program")
async def test_program(request: ProgramTestRequest):
    random.seed(42)
    global erroneous_program
    new_program = ''
    erroneous_program = request.program
    test_cases = request.test_cases
    line_number = request.line_number
    wrong_expression = get_expression_at_line(erroneous_program, line_number)
    wrong_expression=str(wrong_expression)[1:-1].strip()

    try:
        unittest_code = generate_unittest_class(erroneous_program, test_cases)
        # Write the erroneous program to a file
        test_program_file = os.path.join(OUTPUT_DIRECTORY, 'test_program.py')
        with open(test_program_file, 'w') as f:
            f.write(unittest_code)

        # Read the content of the file back into erroneous_program
        with open(test_program_file, 'r') as f:
            erroneous_program = f.read()
        print(erroneous_program)

        
        if not erroneous_program.strip():
            logging.error("Uploaded file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        variables, _ = extract_variables_constants(erroneous_program)
        if not variables:
            logging.error("No variables found in the uploaded program.")
            raise HTTPException(status_code=400, detail="No valid data points found in the uploaded file.")
        
        if len(variables) < 1:
            logging.error("Error: Incorrect number of extracted variables.")
            raise HTTPException(status_code=400, detail="Incorrect number of extracted variables.")
        
        variables_list = list(variables)  # Convert the set to a list
    
        pop = toolbox.population(n=200)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        toolbox.register("evaluate", evalSymbReg, variables=variables_list,line_number=line_number,wrong_expression=wrong_expression )  # Pass the list of variables
        toolbox.register("select", tools.selTournament, tournsize=3)  # Register tournament selection operator
        toolbox.register("mate", gp.cxOnePoint)  # Register crossover operator
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)  # Register mutation operator
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)  # Register mutation operator
        start_time = time.time()
        best_gen = evolution(toolbox, pop, hof, stats,variables, ngen=10)
        end_time = time.time()
        execution_time = end_time - start_time

        best_expr_str = ""
        best_expr_str2 = ""

        if best_gen != -1:
            best_generation = best_gen
            best_individual = hof[0]
            best_expr_str = translate_expr(best_individual, variables_list)  # Pass the list of variables
            best_expr_str2 = expr.parseString(str(best_expr_str))
            print("Best individual infix expression:", best_expr_str2)
            new_program = replace_expression_condition(erroneous_program, line_number, wrong_expression, str(best_expr_str2)[2:-2].strip())
        else:
            print("No solution found within the time limit.")
            best_generation = -1

        return JSONResponse(content={
            "erroneous_program": erroneous_program,
            "new_program": new_program,
            "best_expression": str(best_expr_str2),
            "elapsed_time": execution_time,
            "generation": best_generation
        })
    
    except Exception as e:
        logging.error("Error processing the uploaded file: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8082)
