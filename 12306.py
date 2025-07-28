import requests


url = 'https://kyfw.12306.cn/otn/leftTicket/queryU?leftTicketDTO.train_date=2025-08-01&leftTicketDTO.from_station=GIU&leftTicketDTO.to_station=WGH&purpose_codes=ADULT'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
    'Cookie': '_uab_collina=175370938889697243834269; JSESSIONID=6A4019E7A4E36A0D934F60C9603D213F; BIGipServerotn=1524171018.50210.0000; BIGipServerpassport=837288202.50215.0000; guidesStatus=off; highContrastMode=defaltMode; cursorStatus=off; route=9036359bb8a8a461c164a04f8f50b252; _jc_save_fromStation=%u704C%u5357%2CGIU; _jc_save_toStation=%u65E0%u9521%u4E1C%2CWGH; _jc_save_toDate=2025-07-28; _jc_save_wfdc_flag=dc; _jc_save_fromDate=2025-08-01',
    'referer': 'https://kyfw.12306.cn/otn/leftTicket/init?linktypeid=dc&fs=%E7%81%8C%E5%8D%97,GIU&ts=%E6%97%A0%E9%94%A1%E4%B8%9C,WGH&date=2025-08-01&flag=N,N,Y'
}
response = requests.get(url, headers=headers)
resp_data = response.json().get('data', {}).get('result', [])
for item in resp_data:
    details = item.split('|')
    train_no = details[3]
    from_station = details[6]
    to_station = details[7]
    start_time = details[8]
    arrive_time = details[9]
    duration = details[10]

    print(f"车次: {train_no}, 出发站: {from_station}, 到达站: {to_station}, 出发时间: {start_time}, 到达时间: {arrive_time}, 历时: {duration}")