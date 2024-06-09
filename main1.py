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



# Constants
MAX_GENERATIONS = 5
POPULATION_SIZE = 100
TIMEOUT_SECONDS = 1  # Adjust the timeout as needed
OUTPUT_DIRECTORY = "."

# Define the primitive set for the symbolic regression problem
pset = gp.PrimitiveSet("MAIN", arity=1)  # This will be updated later based on variables
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addEphemeralConstant("rand101", lambda: random.randint(0, 1))

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
    expr = expr.replace("add", "+")
    expr = expr.replace("sub", "-")
    expr = expr.replace("mul", "*")
    expr = expr.replace("rand101", str(random.randint(-10, 10)))
    for i, var in enumerate(variables):
        arg_name = "ARG{}".format(i)
        expr = expr.replace(arg_name, var)
    return expr

async def evalSymbReg(individual, variables, line_number, wrong_expression):
    new_expression = translate_expr(individual, variables)
    new_expression = expr.parseString(str(new_expression))
    erroneous_code = replace_expression(erroneous_program, line_number, wrong_expression, str(new_expression)[2:-2].strip())
    
    try:
        total_tests, failed_tests, successful_tests = await evaluate_program(erroneous_code)
    except Exception as e:
        logging.error(f"Exception during evaluation: {e}")
        return 25,  # Default to all tests failing if there's an error

    num_failed_tests = failed_tests
    return num_failed_tests,

async def async_eaSimple(pop, toolbox, cxpb, mutpb, ngen, halloffame, threshold, stats=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the entire population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = await asyncio.gather(*[toolbox.evaluate(ind) for ind in invalid_ind])
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    if halloffame is not None:
        halloffame.update(pop)
    
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = await asyncio.gather(*[toolbox.evaluate(ind) for ind in invalid_ind])
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
        
        # Replace the current population with the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Check if the best individual meets the threshold
        if halloffame[0].fitness.values[0] <= threshold:
            logging.info(f"Terminating early at generation {gen} as the best individual met the threshold.")
            break

    return pop, logbook
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
    global erroneous_program
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
        
        pset = gp.PrimitiveSet("MAIN", arity=len(variables))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addEphemeralConstant("rand101", lambda: random.randint(0, 2))

        toolbox.register("evaluate", lambda ind: evalSymbReg(ind, variables=variables, line_number=line_number, wrong_expression=wrong_expression))
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        start_time = time.time()
        random.seed(42)

        if len(variables) < 1:
            logging.error("Error: Incorrect number of extracted variables.")
            raise HTTPException(status_code=400, detail="Incorrect number of extracted variables.")
        
        variables_list = list(variables)
        pop = toolbox.population(n=POPULATION_SIZE)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Set a fitness threshold for early termination
        fitness_threshold = 0

        logbook = await async_eaSimple(pop, toolbox, 0.5, 0.1, MAX_GENERATIONS, halloffame=hof, threshold=fitness_threshold, stats=stats, verbose=True)
        execution_time = time.time() - start_time

        metrics_file = OUTPUT_DIRECTORY + "/gcd_evolution_metrics.csv"
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Individuals", "Average Fitness", "Std Fitness", "Min Fitness", "Max Fitness", "Execution Time", "Best Individual"])
            best_individual = hof[0]
            best_expr_str = translate_expr(best_individual, variables_list)
            best_expr_str2 = expr.parseString(str(best_expr_str))
            best_expr_str2 = str(best_expr_str2)[2:-2].strip()
            for gen, record in enumerate(logbook[1:]):
                writer.writerow([record[0]['gen'], record[0]['nevals'], record[0]['avg'], record[0]['std'], record[0]['min'], record[0]['max'], execution_time, best_expr_str2])
                best_generation=record[0]['gen']
            new_program = replace_expression(erroneous_program, line_number, wrong_expression, best_expr_str2)

    
        logging.info("Metrics and best individuals saved successfully.")
        return JSONResponse(content={
        "erroneous_program": erroneous_program,
        "new_program": new_program,
        "best_expression": best_expr_str2,
        "elapsed_time": execution_time,
        "generation": best_generation
    })

    except Exception as e:
        logging.error("Error processing the uploaded file: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)