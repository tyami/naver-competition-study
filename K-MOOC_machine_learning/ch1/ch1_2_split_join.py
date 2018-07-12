## split
# 빈칸을 기준으로 나누기
codes = "python java c++"
code = codes.split()
print(code)

# 특정 문자를 기준으로 나누기
codes = "python, java, c++"
code = codes.split(",")
print(code)

# unpacking (여러 변수로 쪼개기)
example = "cs50.gachon.edu"
subdomain, domain, tId = example.split(".")
print(subdomain,domain,tId)

## join: 여러 문자를 사이에 넣고 연결할 수 있다 !
colors = ["red", "blue", "orange", "green"]
print("".join(colors))
print(", ".join(colors))
print("-".join(colors))
