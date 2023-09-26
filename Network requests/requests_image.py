import requests

url = "https://github.com/zhaoyuzhi/D2HNet/blob/main/assets/data.png"

# Get the requests
res = requests.get(url)
# Use the following command if requests.exceptions.SSLError: HTTPSConnectionPool
# res = requests.get(url, verify=False)

with open('test.jpg', 'wb') as f:
    f.write(res.content)
