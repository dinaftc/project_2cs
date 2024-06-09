class AbsMinusWrongIf2:

    @staticmethod
    def absM(i, j):
        result = 0
        k = 0
        if i <= j:
            k = k + 1
        if k == 0 and i != j:
            result = j - i
        else:
            result = i - j
        return result
