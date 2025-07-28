import requests
import os
from lxml import etree

#第一章链接
url = 'https://www.wenku8.net/novel/2/2254/82462.htm'

while True:
    #请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0'
    }
    #发送请求
    response = requests.get(url, headers=headers)
    #设置编码
    response.encoding = response.apparent_encoding


    e = etree.HTML(response.text)
    print("正在处理：", url)  # 打印当前处理的链接
    # 提取小说内容
    title = e.xpath('//*[@id="title"]/text()')[0]
    print("XPath 匹配结果:", title)
    content = e.xpath('//div[@id="content"]/text()')
    # 提取下一章链接
    next_chapter_link = e.xpath('//div[@id="foottext"]/a[4]/@href')[0]
    url = f'https://www.wenku8.net/novel/2/2254/{next_chapter_link}'
    # 处理内容
    content = '\n'.join(content)
    #输出内容
    #print(title)
    #print(content)

    #保存
    folder_name = "novel"
    file_name = f"{title}.txt"
    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # 保存文件
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(title + '\n')
        f.write(content)
    




