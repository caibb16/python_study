import requests
from lxml import etree

#发送网址
url = 'https://www.wenku8.net/novel/2/2254/82462.htm'
#请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0'
}
#发送请求
response = requests.get(url, headers=headers)
#设置编码
response.encoding = response.apparent_encoding


e = etree.HTML(response.text)
# 提取小说内容
title = e.xpath('//div[@id="title"]/text()')[0]
content = e.xpath('//div[@id="content"]/text()')
# 处理内容
content = '\n'.join(content)
#输出内容
print(title)
print(content)
#保存
with open('弱势角色友崎君.txt', 'w', encoding='utf-8') as f:
    f.write(title + '\n')
    f.write(content)



