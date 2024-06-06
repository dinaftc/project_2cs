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



@app.post("/correct_program")
async def correct_program(request: ProgramTestRequest):
    erroneous_program = request.program
    test_cases = request.test_cases
    if not erroneous_program or not test_cases:
        raise HTTPException(status_code=400, detail="Invalid input")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)