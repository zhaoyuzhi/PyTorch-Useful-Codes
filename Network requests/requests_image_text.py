import requests

def api():
    # define the url for the request
    url = 'http://......'
    # define the parameter for the request
    body_data = {
        "filePath: ......"
    }
    r = requests.get(url = url, params = body_data)
    #r = requests.request(url = url, method = 'get', params = body_data)
    return r

if __name__ == '__main__':

    r = api()

    # if the request is text
    print(r.text)

    # if the request is image
    with open("picture.jpg", "rb") as f:
        f.write(r.content)
        