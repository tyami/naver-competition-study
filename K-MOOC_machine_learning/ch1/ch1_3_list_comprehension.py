## 1
#일반코드
result = []
for i in range(10):
    result.append(i)

result

# list comprehension
result = [i for i in range(10)]
result

# filter
result = [i for i in range(10) if i % 2 == 0]
result

## 2
# for 두개를 이을 수도 있다 !
word_1 = "Hello"
word_2 = "World"
result = [i+j for i in word_1 for j in word_2]
result

## 3
case_1 = ["A", "B", "C"]
case_2 = ["D", "E", "A"]
result = [i+j for i in case_1 for j in case_2]
result

# filter, i와 j가 같지 않을 때만 !
result = [i+j for i in case_1 for j in case_2 if not(i==j)]
result

## 4. 2D list comprehension
words = 'The quick brown fox jumps over the lazy dog'.split()
print(words)
stuff = [[w.upper(), w.lower(), len(w)] for w in words]
stuff
# 이렇게 괄호를 적어주면 2D list가 나옴
