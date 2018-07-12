## Asterisk: *,

# *args: 가변인자. 인자들을 한번에 넘겨줄 때 사용 > tuple
# 활용: 더하기 함수를 만들고 싶은데, 몇 개 변수가 들어올지 모르겠다.
def asterisk_test(a, *args):
    print(a, args)
    print(type(args))
asterisk_test(1,2,3,4,5,6)


# **kargs: 키워드 인자 > dict
def asterisk_test(a, **kargs):
    print(a, kargs)
    print(type(kargs))
asterisk_test(1, b=2, c=3, d=4, e=5, f=6)


# unpacking. 나누어 던져주고 싶을 때
def asterisk_test(a, *args):
    print(a, args[0]) # [0]을 안 붙이면 tuple 안에 들어간 형태로 나옴.
    print(type(args))
asterisk_test(1, (2,3,4,5,6))

def asterisk_test(a, args):
    print(a, *args) # 들어온 tuple 앞에 *를 붙여주면 unpacking
    print(type(args))

asterisk_test(1, (2,3,4,5,6))

# unpacking 예제
a,b,c = ([1,2], [3,4], [5,6])
print(a,b,c)

data = ([1,2], [3,4], [5,6])
print(*data)

def asterisk_test(a,b,c,d):
    print(a,b,c,d)
data = {"b":1, "c":2, "d":3} # 함수 선언 시 인자로 받는다고 선언한 애들만 사용가능. 그 외 에러.
asterisk_test(10, **data)
