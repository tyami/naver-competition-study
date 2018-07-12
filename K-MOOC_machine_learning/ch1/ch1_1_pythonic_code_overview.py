# 인생은 짧으니 파이썬을 쓰세요 !
# 인간의 시간을 효율적으로 쓰기 위해 만들어진 언어

# Split & Join
# List Comprehension
# Enumerate & Zip

# 일반 코드
colors = ["a", "b", "c", "d", "e"]
result = ""

for s in colors:
    result += s

print(result)

# pythonic code
colors = ["a", "b", "c", "d", "e"]
result = "".join(colors)

print(result)
