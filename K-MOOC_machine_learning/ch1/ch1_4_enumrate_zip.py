## Enumerate: list에서 값을 추출할 때, 인덱스 번호를 함께 추출하는 방법
for i, v in enumerate(['tic', 'tac', 'toc']):
    print(i,v)

# enumerate는 문장 내 단어의 위치를 뽑고 싶을 때 사용할 수 있다 > dic 타입으로,
mylist = ["a", "b", "c", "d"]
list(enumerate(mylist))
{i:j for i,j in enumerate('Gacheon University is an academic institute located in South Korea.'.split())}

## Zip: 두 개의 list 값을 병렬적으로 추출
alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']
for a, b in zip(alist, blist):
    print(a,b)

# 각 벡터의 동일한 위치끼리 동일하게 tuple로 묶어줌
a,b,c = zip((1,2,3), (10,20,30), (100,200,300))
a,b,c
[sum(x) for x in zip((1,2,3,), (10,20,30), (100,200,300))] # 벡터합

## Enumerate & zip: 같은 위치의 원소를 뽑고 인덱스도 함께 뽑자 !
alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']

for i, (a,b) in enumerate(zip(alist, blist)):
    print(i,a,b)
