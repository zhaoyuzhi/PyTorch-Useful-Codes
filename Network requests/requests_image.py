import requests

url = "https://github.com/zhaoyuzhi/D2HNet/blob/main/assets/data.png"

# Get the requests
res = requests.get(url)

# Use the following command if requests.exceptions.SSLError: HTTPSConnectionPool
# res = requests.get(url, verify=False)

# Use "try except" commands if urllib3.exceptions.LocationParseError: Failed to parse
# try:
#     res = requests.get(url, verify=False)
# except BaseException:
#     pass

with open('test.jpg', 'wb') as f:
    f.write(res.content)
