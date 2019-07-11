# -*- coding: utf-8 -*-
def operation_add(num1, num2):
    return num1 + num2

def operation_sub(num1, num2):
    return num1 - num2

def evaluate(symbols):
    """
    expects an array of symbols
    each symbos is a number between 0 -> 11
    digits are => 0:9
    addition => 10
    subtraction => 11
    """
    operands = [0]
    operand_idx = 0
    operations = []
    # store the sign of the first number as an operation
    if symbols[0] < 11:
        operations.append(operation_add)
    else:
        operations.append(operation_sub)

    for symbol in symbols:
        if symbol < 10:
            # this means it's a digit
            # add it to the least significant place in the operand
            operands[operand_idx] = operands[operand_idx] * 10 + symbol
        elif symbol == 10:
            operations.append(operation_add)
            operands.append(0)
            operand_idx += 1
        else:
            operations.append(operation_sub)
            operands.append(0)
            operand_idx += 1

    
    result = 0
    for operand, operation in zip(operands, operations):
        result = operation(result, operand)
    return result
        