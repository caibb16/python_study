# 从小到大计算Sn=∑(1/(j^2-1)), j从2到N
j = 2
N = int(input("请输入一个正整数N："))
Sn = 0
while (j <= N ):
    Sn = Sn + 1 / (j * j - 1)
    j += 1  
print(Sn)

