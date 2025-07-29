import requests

station_url = 'https://kyfw.12306.cn/otn/resources/js/framework/station_name.js?station_version=1.9346'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0'
}

response = requests.get(station_url, headers=headers)
# 提取数据部分
station_data = response.text[response.text.find('=') :-1]  
station_data = station_data.split('|')

num = len(station_data)//10     # 计算车站数量
# 创建车站字典
station_dict = {}
for i in range(num):
    station_info = station_data[i*10:i*10+10]
    station_dict[station_info[1]] = station_info[2]

print(station_dict)  # 打印车站字典
