# https://github.com/TEAMLAB-Lecture/AI-python-connect/tree/master/lab_assignments/lab_1
def vector_size_check(*vector_variables):
    return all(len(vector_variables[0]) == x for x in [len(v) for v in vector_variables[1:]])
print(vector_size_check([1,2,3], [2,3,4], [5,6,7]))
print(vector_size_check([1, 3], [2,4], [6,7]))
print(vector_size_check([1, 3, 4], [4], [6,7]))

# all: 괄호 안 리스트 원소가 모두 True일 때  True
all([True, True, False])
all([True, True, True])

def vector_addition(*vector_variables):
    return [sum(element) for element in zip(*vector_variables)]
vector_addition([1,2,3], [2,3,4], [5,6,7])
vector_addition([1,2,3,4], [2,3,4,5])


def vector_subtraction(*vector_variables):
    if vector_size_check(*vector_variables) == False:
        raise ArithmeticError

    return [element[0]*2-sum(element) for element in zip(*vector_variables)]
vector_subtraction([1,2,3,4], [2,3,4,5])


def scalar_vector_product(alpha, vector_variable):
    return [alpha*x for x in vector_variable]
scalar_vector_product(4, [2,3,4,5])


for a in zip([[2,3],[2,1]], [[4,5],[7,8]]):
    print(a)


def matrix_size_check(*matrix_variables):
    print(len(matrix_variables[0]))
    return all([all(len(matrix_variables[0]) == len(matrix) for matrix in matrix_variables[1:]), # row
                all(len(matrix_variables[0][0]) == len(vector) for vector in matrix_variables[1:][0])]) # col
matrix_x = [[2, 2], [2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4, 3], [5, 3, 2]]
matrix_w = [[2, 5], [1, 1], [2, 2]]
print (matrix_size_check(matrix_x, matrix_y, matrix_z)) # Expected value: False
print (matrix_size_check(matrix_y, matrix_z)) # Expected value: False
print (matrix_size_check(matrix_x, matrix_w)) # Expected value: True

def is_matrix_equal(*matrix_variables):
    return all(e == f for e in zip(matrix_variables[0]) for f in zip(matrix_variables[1]))
print (is_matrix_equal(matrix_x, matrix_y, matrix_y, matrix_y)) # Expected value: False
print (is_matrix_equal(matrix_x, matrix_x)) # Expected value: True

def matrix_addition(*matrix_variables):
    if matrix_size_check(*matrix_variables) == False:
        raise ArithmeticError
    return [[sum(element) for element in zip(*matrix)] for matrix in zip(*matrix_variables)]
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]
print (matrix_addition(matrix_x, matrix_y)) # Expected value: [[4, 7], [4, 3]]
print (matrix_addition(matrix_x, matrix_y, matrix_z)) # Expected value: [[6, 11], [9, 6]]

def matrix_subtraction(*matrix_variables):
    if matrix_size_check(*matrix_variables) == False:
        raise ArithmeticError
    return [[element[0]*2-sum(element) for element in zip(*matrix)] for matrix in zip(*matrix_variables)]
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
print (matrix_subtraction(matrix_x, matrix_y)) # Expected value: [[4, 7], [4, 3]]

def matrix_transpose(matrix_variable):
    return [[e for e in col] for col in zip(*matrix)]
matrix = [[2, 5], [2, 1]]
matrix_transpose(matrix)

def scalar_matrix_product(alpha, matrix_variable):
    return [[alpha*e for e in row] for row in matrix]
scalar_matrix_product(3, matrix)

def is_product_availability_matrix(matrix_a, matrix_b):
    # a의 col == b의 row
    return len(matrix_a[0]) == len(matrix_b)
matrix_a = [[2, 2], [2, 2]]
matrix_b = [[2, 5], [2, 1]]
matrix_c = [[2, 5], [2, 1], [4, 7]]
matrix_d = [[2, 5, 4], [2, 1, 7]]
is_product_availability_matrix(matrix_a, matrix_b)
is_product_availability_matrix(matrix_a, matrix_c)
is_product_availability_matrix(matrix_c, matrix_d)

def matrix_product(matrix_a, matrix_b):
    if is_product_availability_matrix(matrix_a, matrix_b) == False:
        raise ArithmeticError
    # a의 row X b의 col
    return [[sum(row_a*col_b for row_a, col_b in zip(row_a, col_b)) for col_b in zip(*matrix_b)] for row_a in matrix_a]
matrix_product(matrix_a, matrix_b)
