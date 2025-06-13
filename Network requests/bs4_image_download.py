import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time


def search_and_download_images(keyword, num_images, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置 Chrome 选项
    chrome_options = Options()
    chrome_options.add_argument('--headless')

    # 设置 ChromeDriver 路径
    service = Service('./Network requests/chromedriver-win64/chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # 构建 Google 图像搜索 URL
    search_url = f"https://www.google.com/search?q={keyword}&tbm=isch"
    driver.get(search_url)

    # 模拟滚动页面以加载更多图片
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # 获取页面源代码
    page_source = driver.page_source
    driver.quit()

    # 解析页面源代码
    soup = BeautifulSoup(page_source, 'html.parser')
    img_tags = soup.find_all('img')

    # 下载图片
    count = 0
    for img_tag in img_tags:
        if count >= num_images:
            break
        try:
            img_url = img_tag['src']
            if img_url.startswith('data:'):
                continue
            response = requests.get(img_url, stream=True)
            if response.status_code == 200:
                file_name = os.path.join(output_dir, f"{keyword}_{count}.jpg")
                with open(file_name, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                count += 1
        except Exception as e:
            print(f"Error downloading image: {e}")


if __name__ == "__main__":
    keyword = input("请输入搜索关键词: ")
    num_images = int(input("请输入需要爬取的图像数量: "))
    output_dir = "downloaded_images"
    search_and_download_images(keyword, num_images, output_dir)
