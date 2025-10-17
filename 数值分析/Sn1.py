
N = int(input("请输入一个正整数N："))
j = N
Sn = 0
while (j >= 2):
    Sn = Sn + 1 / (j * j - 1)
    j -= 1  
print(Sn)