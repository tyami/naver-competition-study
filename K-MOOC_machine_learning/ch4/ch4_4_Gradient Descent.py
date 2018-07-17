import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 11, 1)
f_x = x **2

plt.plot(x, f_x)
plt.show()


# gradient descent
x_new = 10
derivate = [x_new]
y = [x_new**2]
learning_rate = 0.1
for i in range(100):
    old_value = x_new
    x_new = old_value - learning_rate * 2 * old_value
    derivate.append(x_new)
    y.append(x_new ** 2)

plt.plot(x, f_x)
plt.scatter(np.sqrt(y), y)
plt.show()

# 미분함수 찾기: http://www.wolframalpha.com/input/?i=derivative+x+sin(x%5E2)+%2B+1


# 초기값이 중요하다.
def sin_function(x):
    return x * np.sin(x ** 2) + 1

def derivate_f(x):
    return np.sin(x**2) + 2*(x**2)*np.cos(x**2)

x = np.arange(-3,3,0.001)
f_x = sin_function(x)

x_new = 2.6 # -2를 넣어보자
derivate = [x_new]
y = [sin_function(x_new)]
learning_rate = 0.01
for i in range(100):
    old_value = x_new
    x_new = old_value - learning_rate * derivate_f(old_value)
    derivate.append(x_new)
    y.append(sin_function(x_new))

plt.plot(x, f_x)
plt.scatter(derivate, y)
plt.show()
