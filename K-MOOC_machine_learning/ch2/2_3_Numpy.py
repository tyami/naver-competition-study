## Python은 C보다 훨~씬 느리다.

# 호출
import numpy as np

# ndarray: numpy의 기본 단위
test_array = np.array(["1","4",5,7],float)
print(test_array)
# dynamic typing 지원 X: numpy는 하나의 데이터타입만 변수에 들어갈 수 있다.
# 덕분에 속도가 빠름 !
type(test_array[3])
# float64: 한 element의 메모리 공간이 64bit
# 딥러닝 시에는 float8, 32... 같이 줄여서 메모리 효율 늘리기.

test_array = np.array([[1,2,3],[4,5,6],[7,8,9]], float)
# shape: numpy array의 형태: tuple로 output
print(test_array.shape)
# 1D: col
# 2D: row X col
# 3D: depth X row X col
# ...

# dtype: 전체 데이터 dtype
print(test_array.dtype)

# ndim: dim 수
print(test_array.ndim)

# size: 전체 element 수
print(test_array.size)


# reshape: shape 형태를 바꿀 때> nData 만 맞추면 됨
test_array = np.array([[2,2,],[3,2],[6,5],[1,6]])
test_array
test_array.reshape(1,8)
test_array.reshape(-1,4) # -1: 정확한 개수는 모르고 나머지에 맞추겠다.

# flatten: reshape으로 해도 상관없지만, 고차원 -> 1차원으로 바꿔줄 때
test_array.flatten()

## tolist
test_array.tolist()


## indexing & slicing
test_array[0][0]
test_array[0,0]
test_array[1:]
test_array[-1]
test_array[-2:]


## creation function
np.arange(30)
np.arange(0, 5, 0.5)
np.arange(30).reshape(5,-1)


np.zeros(shape=(10,), dtype=np.int8)
np.ones(shape=(10,), dtype=np.int8)
np.empty(shape=(10,), dtype=np.int8)

## something_like
np.zeros_like(test_array)
np.ones_like(test_array)
np.identity(5)
np.eye(3),
np.eye(3,5,k=2)
np.diag(np.arange(9).reshape(3,3))


## random sampling
np.random.uniform(0,1,10).reshape(2,5)
np.random.normal(0,1,10).reshape(2,5)

# sum
test_array
np.sum(test_array)
test_array.sum()
np.sum(test_array,0)
test_array.sum(1)

# concatenate. vstack, hstack
a = np.array([1,2,3])
b = np.array([4,5,6])
np.vstack((a,b))
np.hstack((a,b))
np.concatenate((a,b), axis=0)
np.concatenate((a,b))


## Operations btw. arrays
a = np.array([[1,2,3],[4,5,6]])
a
a + a
a - a
a * a # element 곱
a.dot(a.transpose()) # 내적


## broadcasting: shape이 다른 배열간 연산 지원
a
scalar = 3

a + scalar

# scalar-vector 뿐아니라 vector-matrix끼리도 broadcasting
a = np.array([1,2,3])
b = np.array([[1,2,3], [4,5,6]])
a
b
a + b

# operation 속도: numpy > list comprehension > for
# concatenate 속도: numpy < list (numpy는 붙어있는 메모리를 searching 해야 함)

## comparison
a = np.arange(10)
a

a > 5

np.any(a>5), np.any(a>10) # OR
np.all(a>5), np.all(a<10) # AND

np.logical_and(a>0, a<3)
# np.logical_or 이나 np.logical_not도 있음

# np.where
np.where(a>0, 3,2)
# a>0 조건에 대해서 True일 경우 3을, False 일 경우 2를 반환

np.where(a>0)
# 아무것도 쓰지 않으면 index 값 return

# NaN, Inf
a = np.array([1, np.NaN, np.Inf])
np.isnan(a)
np.isinf(a)

# argmax, argmin: numpy에서는  for문을 쓰지 말자 !!!
a = np.array([1,2,3,4,5,6,38,29,-3, 0])
a.size
np.argmax(a) # 최대값의 index (38)
np.argmin(a)

b = np.array([[1,2,3],[4,5,75],[7,8,9]])
np.argmax(b, axis=0), np.argmin(b, axis=1)

## boolean index
test_array = np.array([1,4,0,2,3,4,6,2], float)
condition = test_array > 3
test_array[condition]
condition.astype(np.int) # binary 형태로 바꿔줄 수도 있다

## fancy index
a = np.array([2,4,6,8], float)
b = np.array([0,0,0,1,3,2,1], int)
a[b]
a.take(b) # take 함수를 쓰는게 fancy index 라는걸 알아보기 쉽다. > 권장

a = np.array([[2,4,6,8],[1,3,5,7]], float)
r = np.array([0,0,1])
c = np.array([0,1,0])
a[r,c]
