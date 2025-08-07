import requests

train_date = '2025-08-01'  # 设置查询日期
from_station = '灌南'  # 出发站
to_station = '无锡'  # 到达站

# 获取车站信息
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


#获取票务信息

from_station_code = station_dict.get(from_station)
to_station_code = station_dict.get(to_station)
print(f"出发站代码: {from_station_code}, 到达站代码: {to_station_code}")
info_url = f'https://kyfw.12306.cn/otn/leftTicket/queryU?leftTicketDTO.train_date={train_date}&leftTicketDTO.from_station={from_station_code}&leftTicketDTO.to_station={to_station_code}&purpose_codes=ADULT'
print(f"查询URL: {info_url}")
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
    'Cookie': '_uab_collina=175379357377683288091176; JSESSIONID=B67D61D854D8D27DAD69359F5B940A09; route=9036359bb8a8a461c164a04f8f50b252; BIGipServerotn=1406730506.50210.0000; _jc_save_fromStation=%u704C%u5357%2CGIU; _jc_save_toStation=%u65E0%u9521%u4E1C%2CWGH; _jc_save_fromDate=2025-08-01; _jc_save_toDate=2025-07-29; _jc_save_wfdc_flag=dc; guidesStatus=off; highContrastMode=defaltMode; cursorStatus=off; BIGipServerpassport=937951498.50215.0000'  # 替换为实际的JSESSIONID
}
response = requests.get(info_url, headers=headers)
resp_data = response.json().get('data', {}).get('result', [])
for item in resp_data:
    details = item.split('|')
    train_no = details[3]
    start_time = details[8]
    arrive_time = details[9]
    duration = details[10]
    no_seat = details[26]  # 无座
    second_class_seat = details[30]  # 二等座
    first_class_seat = details[31]  # 一等座
    

    print(f"车次: {train_no}, 出发站: {from_station}, 到达站: {to_station}, 出发时间: {start_time}, 到达时间: {arrive_time}, 历时: {duration},一等座： {first_class_seat}, 二等座: {second_class_seat}, 无座: {no_seat}")