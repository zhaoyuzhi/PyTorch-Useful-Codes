import requests

url = "https://github.com/zhaoyuzhi/D2HNet/blob/main/assets/data.png"

res = requests.get(url)

with open('test.jpg', 'wb') as f:
    f.write(res.content)
