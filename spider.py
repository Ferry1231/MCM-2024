# 导入request库和pandas库
import requests
import pandas as pd

url = 'https://en.wikipedia.org/wiki/2017_French_Open_%E2%80%93_Men%27s_Singles'
headers = {
     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
}
params = {
     'action': 'render'
}

# 发送请求，获取响应对象
response = requests.get(url, headers=headers, params=params)

# 检查响应状态码
if response.status_code == 200:
     print('请求成功')
else:
     print('请求失败')

html = response.text
tables = pd.read_html(html)
seeds = tables[0]
print(seeds.head())
seeds.to_csv('seeds.csv', index=False)