## Collection: Stack, Queue 이런거

## deque: stack, queue 구현 시 사용
# list에 비해 효율적 (빠른) 자료 저장방식이다.
from collections import deque

deque_list = deque()
for i in range(5):
    deque_list.append(i)
print(deque_list)

deque_list.appendleft(10)
print(deque_list)


## OrderedDict: dict와 달리, 데이터 입력 순서대로 dict를 반환
d = {}
d['x'] = 100
d['y'] = 200
d['z'] = 300
d['l'] = 500
for k,v in d.items():
    print(k,v)

from collections import OrderedDict
d = OrderedDict()
d['x'] = 100
d['y'] = 200
d['z'] = 300
d['l'] = 500
for k,v in d.items():
    print(k,v)

# ordereddict의 사용: dict의 value 또는 키 값을 기준으로 sort할때
for k,v in OrderedDict(sorted(d.items(), key=lambda t:t[0])).items():
    print(k, v)
for k,v in OrderedDict(sorted(d.items(), key=lambda t:t[1])).items():
    print(k, v)

## defaultdict: dict type의 기본값 지정
from collections import defaultdict

d = defaultdict(object) # default dict 생성
d = defaultdict(lambda: 0) # 기본 value: 0 으로 설정
print(d)
print(d["first"])

# 응용은 이렇게
text = "travel goal journey travel travel date calendar trip calendar".lower().split()

from collections import defaultdict
word_count = defaultdict(object)
word_count = defaultdict(lambda:0)
for word in text:
    word_count[word] += 1

for i, v in OrderedDict(sorted(
        word_count.items(), key=lambda t: t[1],
        reverse=True)).items():
    print(i,v)

# 같은 걸 dict로 구현하려면?
word_count = dict()
for word in text:
    word_count[word] = word_count.get(word, 0) + 1

for i, v in OrderedDict(sorted(
        word_count.items(), key=lambda t: t[1],
        reverse=True)).items():
    print(i,v)

## Counter: Sequence type의 data element 갯수를 dict 형태로 반환
from collections import Counter

c = Counter() # a new, empty counter
c = Counter('gallahad') # a new counter from an iterable
print(c)
