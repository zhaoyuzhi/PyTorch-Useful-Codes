import os
import time
import random
import requests
import io
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import argparse

class GoogleImageScraper:
    def __init__(
        self, 
        save_path: str = "./downloaded_images",
        driver_path: str = None,
        headless: bool = True,
        min_resolution: tuple = (0, 0),
        max_resolution: tuple = (9999, 9999),
        max_mb: float = 10.0,
        sleep_range: tuple = (1, 3)
    ):
        """
        初始化图像爬虫
        
        Args:
            save_path: 保存图像的根目录
            driver_path: 手动指定ChromeDriver路径（可选）
            headless: 是否使用无头模式
            min_resolution: 最小图像分辨率要求 (宽度, 高度)
            max_resolution: 最大图像分辨率要求 (宽度, 高度)
            max_mb: 最大图像大小(MB)
            sleep_range: 请求间隔时间范围 (最小值, 最大值)
        """
        self.save_path = save_path
        self.driver_path = driver_path
        self.headless = headless
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_mb = max_mb
        self.sleep_range = sleep_range
        self.driver = None
        
    def _setup_driver(self):
        """初始化WebDriver"""
        chrome_options = Options()
        
        # 无头模式设置
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # 其他浏览器选项
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # 初始化驱动
        if self.driver_path:
            self.driver = webdriver.Chrome(
                executable_path=self.driver_path, 
                options=chrome_options
            )
        else:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), 
                options=chrome_options
            )
        
        # 绕过检测
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        print("[INFO] WebDriver初始化成功")
    
    def _random_sleep(self):
        """随机休眠，避免请求过于频繁"""
        sleep_time = random.uniform(self.sleep_range[0], self.sleep_range[1])
        time.sleep(sleep_time)
    
    def _scroll_to_end(self):
        """滚动到页面底部以加载更多图像"""
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self._random_sleep()
    
    def _click_load_more(self):
        """点击"加载更多"按钮"""
        try:
            load_more_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".mye4qd"))
            )
            load_more_button.click()
            print("[INFO] 点击了加载更多按钮")
            self._random_sleep()
            return True
        except Exception as e:
            print(f"[INFO] 未找到加载更多按钮或点击失败: {e}")
            return False
    
    def fetch_image_urls(self, query: str, max_images: int) -> list:
        """
        搜索关键词并获取图像URL列表
        
        Args:
            query: 搜索关键词
            max_images: 最大获取数量
        
        Returns:
            图像URL列表
        """
        if not self.driver:
            self._setup_driver()
        
        # 构建搜索URL
        search_url = f"https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={query}&oq={query}&gs_l=img"
        
        # 加载页面
        self.driver.get(search_url)
        print(f"[INFO] 加载搜索页面: {query}")
        
        image_urls = set()
        last_count = 0
        attempts = 0
        max_attempts = 5  # 尝试加载更多的最大次数
        
        while len(image_urls) < max_images and attempts < max_attempts:
            # 滚动页面加载更多内容
            self._scroll_to_end()
            
            # 尝试点击"加载更多"按钮
            if len(image_urls) > 0 and len(image_urls) % 100 == 0:
                self._click_load_more()
            
            # 获取所有可见的图像元素
            thumbnail_elements = self.driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
            
            # 如果没有新图像加载，增加尝试次数
            if len(thumbnail_elements) == last_count:
                attempts += 1
                print(f"[INFO] 没有新图像加载，尝试次数: {attempts}/{max_attempts}")
            else:
                attempts = 0
            
            last_count = len(thumbnail_elements)
            print(f"[INFO] 当前找到 {len(thumbnail_elements)} 个图像元素")
            
            # 遍历并点击缩略图获取大图URL
            for i, img in enumerate(thumbnail_elements):
                if len(image_urls) >= max_images:
                    break
                
                try:
                    # 滚动到图像元素
                    self.driver.execute_script("arguments[0].scrollIntoView();", img)
                    self._random_sleep()
                    
                    # 点击缩略图
                    img.click()
                    self._random_sleep()
                    
                    # 等待大图加载
                    WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "img.n3VNCb"))
                    )
                    
                    # 获取大图URL
                    actual_images = self.driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
                    for actual_image in actual_images:
                        src = actual_image.get_attribute("src")
                        if src and src.startswith("http") and "encrypted-tbn0.gstatic.com" not in src:
                            image_urls.add(src)
                            print(f"[INFO] 找到图像 #{len(image_urls)}: {src[:50]}...")
                            break
                except Exception as e:
                    print(f"[INFO] 处理图像元素时出错: {e}")
                    continue
            
            # 如果达到最大尝试次数仍未找到更多图像，退出循环
            if attempts >= max_attempts:
                print("[INFO] 达到最大尝试次数，停止加载更多图像")
                break
        
        return list(image_urls)[:max_images]
    
    def download_image(self, url: str, folder_path: str, counter: int) -> bool:
        """
        下载并保存单个图像
        
        Args:
            url: 图像URL
            folder_path: 保存文件夹路径
            counter: 图像序号
        
        Returns:
            是否下载成功
        """
        try:
            # 获取图像内容，设置超时时间
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            # 检查文件大小
            if "content-length" in response.headers:
                size_mb = int(response.headers["content-length"]) / (1024 * 1024)
                if size_mb > self.max_mb:
                    print(f"[WARNING] 图像过大 ({size_mb:.2f}MB)，跳过: {url[:50]}...")
                    return False
            
            # 处理图像
            image_content = response.content
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file)
            
            # 检查图像分辨率
            width, height = image.size
            if (width < self.min_resolution[0] or height < self.min_resolution[1] or
                width > self.max_resolution[0] or height > self.max_resolution[1]):
                print(f"[WARNING] 图像分辨率不符合要求 ({width}x{height})，跳过: {url[:50]}...")
                return False
            
            # 保存图像
            file_path = os.path.join(folder_path, f"image_{counter}.jpg")
            image.save(file_path, "JPEG", quality=90)
            print(f"[SUCCESS] 下载并保存图像 #{counter}: {file_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 下载图像失败: {url[:50]}... 错误: {e}")
            return False
    
    def search_and_download(self, query: str, max_images: int = 10) -> int:
        """
        搜索关键词并下载图像
        
        Args:
            query: 搜索关键词
            max_images: 最大下载数量
        
        Returns:
            成功下载的图像数量
        """
        # 创建保存文件夹
        keyword_folder = '_'.join(query.lower().split(' '))
        save_folder = os.path.join(self.save_path, keyword_folder)
        os.makedirs(save_folder, exist_ok=True)
        print(f"[INFO] 图像将保存至: {save_folder}")
        
        try:
            # 获取图像URL
            image_urls = self.fetch_image_urls(query, max_images)
            print(f"[INFO] 找到 {len(image_urls)} 个图像URL")
            
            # 下载图像
            success_count = 0
            for i, url in enumerate(image_urls):
                if success_count >= max_images:
                    break
                
                if self.download_image(url, save_folder, i):
                    success_count += 1
                
                # 每下载10张图像显示进度
                if (i + 1) % 10 == 0 or i == len(image_urls) - 1:
                    print(f"[INFO] 进度: {i + 1}/{len(image_urls)}，已保存: {success_count}/{max_images}")
            
            print(f"[INFO] 下载完成，共保存 {success_count} 张图像")
            return success_count
            
        except Exception as e:
            print(f"[ERROR] 程序执行失败: {e}")
            return 0
        finally:
            # 关闭浏览器
            if self.driver:
                self.driver.quit()
                print("[INFO] WebDriver已关闭")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="Google图像爬虫")
    parser.add_argument("--keyword", type=str, required=True, help="搜索关键词")
    parser.add_argument("--count", type=int, default=10, help="下载图像数量")
    parser.add_argument("--path", type=str, default="./downloaded_images", help="保存路径")
    parser.add_argument("--driver", type=str, default=None, help="ChromeDriver路径")
    parser.add_argument("--show", action="store_true", help="显示浏览器窗口")
    parser.add_argument("--min-width", type=int, default=0, help="最小图像宽度")
    parser.add_argument("--min-height", type=int, default=0, help="最小图像高度")
    parser.add_argument("--max-size", type=float, default=10.0, help="最大图像大小(MB)")
    
    args = parser.parse_args()
    
    # 初始化爬虫
    scraper = GoogleImageScraper(
        save_path=args.path,
        driver_path=args.driver,
        headless=not args.show,
        min_resolution=(args.min_width, args.min_height),
        max_mb=args.max_size
    )
    
    # 执行搜索和下载
    scraper.search_and_download(args.keyword, args.count)

if __name__ == "__main__":
    main()
