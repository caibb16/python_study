import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Titanic_data/train.csv', index_col='PassengerId')

# 修改配置添加中文字体
plt.rcParams['font.sans-serif'].insert(0, 'SimHei')
plt.rcParams['axes.unicode_minus'] = False

# 定制画布
plt.figure(figsize=(16, 12), dpi=200)

# 遇难和获救人数分布
plt.subplot(3, 4, 1)
ser = df.Survived.value_counts()
ser.plot(kind='bar', color=['#BE3144', '#3A7D44'])
plt.xticks(rotation=0)
plt.title('图1.获救情况分布')
plt.ylabel('人数')
plt.xlabel('')
for i, v in enumerate(ser):
    plt.text(i, v, v, ha='center')

# 客舱等级人数分布
plt.subplot(3, 4, 2)
ser = df.Pclass.value_counts().sort_index()
ser.plot(kind='bar', color=['#FA4032', '#FA812F', '#FAB12F'])
plt.xticks(rotation=0)
plt.ylabel('人数')
plt.xlabel('')
plt.title('图2.客舱等级分布')
for i, v in enumerate(ser):
    plt.text(i, v, v, ha='center')

# 性别人数分布
plt.subplot(3, 4, 3)
ser = df.Sex.value_counts()
ser.plot(kind='bar', color=['#16404D', '#D84040'])
plt.xticks(rotation=0)
plt.ylabel('人数')
plt.xlabel('')
plt.title('图3.性别分布')
for i, v in enumerate(ser):
    plt.text(i, v, v, ha='center')

# 登船港口人数分布
plt.subplot(3, 4, 4)
ser = df.Embarked.value_counts()
ser.plot(kind='bar', color=['#FA4032', '#FA812F', '#FAB12F'])
plt.xticks(rotation=0)
plt.ylabel('人数')
plt.xlabel('')
plt.title('图4.登船港口分布')
for i, v in enumerate(ser):
    plt.text(i, v, v, ha='center')

# 乘客年龄箱线图
plt.subplot(3, 4, 5)
df.Age.plot(kind='box', showmeans=True, notch=True)
plt.title('图5.乘客年龄情况')

# 船票价格箱线图
plt.subplot(3, 4, 6)
df.Fare.plot(kind='box', showmeans=True, notch=True)
plt.title('图6.船票价格情况')

# 不同客舱等级遇难和幸存人数分布
plt.subplot(3, 4, (7, 8))
s0 = df[df.Survived == 0].Pclass.value_counts()
s1 = df[df.Survived == 1].Pclass.value_counts()
temp = pd.DataFrame({'遇难': s0, '幸存': s1})
pcts = temp.div(temp.sum(axis=1), axis=0)
temp.plot(ax=plt.gca(), kind='bar', stacked=True, color=['#BE3144', '#3A7D44'])
for i, idx in enumerate(temp.index):
    plt.text(i, temp.at[idx, '遇难'] // 2, f'{pcts.at[idx, "遇难"]:.2%}', ha='center', va='center')
    plt.text(i, temp.at[idx, '遇难'] + temp.at[idx, '幸存'] // 2, f'{pcts.at[idx, "幸存"]:.2%}', ha='center', va='center')
plt.xticks(rotation=0)
plt.xlabel('')
plt.title('图7.不同客舱等级幸存情况')

# 不同性别遇难和幸存人数分布
plt.subplot(3, 4, (9, 10))
s0 = df[df.Survived == 0].Sex.value_counts()
s1 = df[df.Survived == 1].Sex.value_counts()
temp = pd.DataFrame({'遇难': s0, '幸存': s1})
pcts = temp.div(temp.sum(axis=1), axis=0)
temp.plot(ax=plt.gca(), kind='bar', stacked=True, color=['#BE3144', '#3A7D44'])
for i, idx in enumerate(temp.index):
    plt.text(i, temp.at[idx, '遇难'] // 2, f'{pcts.at[idx, "遇难"]:.2%}', ha='center', va='center')
    plt.text(i, temp.at[idx, '遇难'] + temp.at[idx, '幸存'] // 2, f'{pcts.at[idx, "幸存"]:.2%}', ha='center', va='center')
plt.xticks(rotation=0)
plt.xlabel('')
plt.title('图8.不同性别幸存情况')

# 不同登船港口遇难和幸存人数分布
plt.subplot(3, 4, (11, 12))
s0 = df[df.Survived == 0].Embarked.value_counts()
s1 = df[df.Survived == 1].Embarked.value_counts()
temp = pd.DataFrame({'遇难': s0, '幸存': s1})
pcts = temp.div(temp.sum(axis=1), axis=0)
temp.plot(ax=plt.gca(), kind='bar', stacked=True, color=['#BE3144', '#3A7D44'])
for i, idx in enumerate(temp.index):
    plt.text(i, temp.at[idx, '遇难'] // 2, f'{pcts.at[idx, "遇难"]:.2%}', ha='center', va='center')
    plt.text(i, temp.at[idx, '遇难'] + temp.at[idx, '幸存'] // 2, f'{pcts.at[idx, "幸存"]:.2%}', ha='center', va='center')
plt.xticks(rotation=0)
plt.xlabel('')
plt.title('图9.不同登船港口幸存情况')

plt.tight_layout()
plt.show()