## Lambda: 함수 이름 없이, 함수처럼 사용할 수 있는 익명함수

# 이런 식을
def f(x,y):
    return x+y
print(f(1,4))

# 이렇게 바꿀 수 있다
f = lambda x,y: x+y
print(f(1,4))

f = lambda x: x**2
print(f(3))


## Map function: sequence 자료형 각 원소에 동일한 function 적용
ex = [1,2,3,4,5]
f  = lambda x: x**2
print(list(map(f,ex)))
# ex의 각 원소에 제곱 함수를 적용.
map(f,ex)
# 앞에 list를 붙여줘야 결과가 나오고, 아니면 위치값이 나옴.
for i in map(f,ex):
    print(i) # 아니면 이렇게 for문


## Reduce: list에 똑같은 함수를 적용해서 통합.
from functools import reduce
print(reduce(lambda x, y: x+y, [1,2,3,4,5]))

def factorial(n):
    return reduce(lambda x, y: x*y, range(1, n+1))

factorial(5)
