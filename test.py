import unittest
class AbsMinusWrongIf2:
    @staticmethod
    def absM(i, j):
        result = 0
        k = 0
        if i <= j:
            k = k + 1
        if (0 > result):  # (k == 1 and i != j)
            result = j - i
        else:
            result = i - j
        return result
    
# Define a test class inheriting from unittest.TestCase
class TestAddFunction(unittest.TestCase):
    def test_0(self):
        self.assertEqual(AbsMinusWrongIf2.absM(1, 2), 1)  

    def test_1(self):
        self.assertEqual(AbsMinusWrongIf2.absM(5, 6), 1)  

    def test_2(self):
        self.assertEqual(AbsMinusWrongIf2.absM(5, 8), 3)  

    def test_3(self):
        self.assertEqual(AbsMinusWrongIf2.absM(1, 6), 5)  
        
    def test_4(self):
        self.assertEqual(AbsMinusWrongIf2.absM(1, 8), 7)  
    
    def test_5(self):
        self.assertEqual(AbsMinusWrongIf2.absM(10, 2), 8)  
    
    def test_6(self):
        self.assertEqual(AbsMinusWrongIf2.absM(10, 6), 4)  
        
    def test_7(self):
        self.assertEqual(AbsMinusWrongIf2.absM(5, 2), 3)

    def test_8(self):
        self.assertEqual(AbsMinusWrongIf2.absM(10, 8), 2)        

if __name__ == '__main__':
    unittest.main()