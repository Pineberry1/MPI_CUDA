import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 群盘的URL
URL = 'https://rec.ustc.edu.cn/group/58633880/disk'  # 替换为实际的群盘URL

# 文件夹名称（根据页面上的按钮文本或属性）
FOLDER_NAME = '作业相关'  # 替换为实际的文件夹名称

# 保存最新修改时间的文件
LAST_MOD_TIME_FILE = 'last_mod_time.txt'
# 从浏览器中复制的Cookie
cookies = [
    {'name': '_ga', 'value': 'GA1.3.1923622707.1721981224', 'domain': 'rec.ustc.edu.cn'},
    {'name': '_ga_VR0TZSDVGE', 'value': 'GS1.3.1721981224.1.0.1721981224.0.0.0', 'domain': 'rec.ustc.edu.cn'},
    {'name': 'sduuid', 'value': 'f806c70e53ff1412ea50ccd0aebc4c5b', 'domain': 'rec.ustc.edu.cn'},
    {'name': 'Rec-Storage', 'value': 'moss', 'domain': 'rec.ustc.edu.cn'},
    {'name': 'Rec-Token', 'value': '81aceb98f80245d1aa4b407aa75af90d', 'domain': 'rec.ustc.edu.cn'},
    {'name': 'Rec-RefreshToken', 'value': '{%22refresh_token%22:%22b7c215d7138741168c1e0f4285c91a70%22%2C%22token_expire_time%22:%222025-03-12%2021:32:32%22}', 'domain': 'rec.ustc.edu.cn'}
]
# 初始化Selenium WebDriver
def init_driver():
    options = webdriver.EdgeOptions()
    #options.add_argument('--headless')  # 无头模式，不显示浏览器窗口
    driver = webdriver.Edge(options=options)
    return driver

# 加载Cookie
def load_cookie(driver):
    driver.get(URL)  # 先访问网站，确保域名一致
    time.sleep(2)  # 等待页面加载
    driver.add_cookie(cookies)  # 添加Cookie
    driver.refresh()  # 刷新页面，使Cookie生效
# 进入文件夹
def enter_folder(driver, folder_name):
    # 定位文件夹按钮并点击
    folder_button = driver.find_element(By.XPATH, f'//button[text()="{folder_name}"]')  # 替换为实际的按钮定位方式
    folder_button.click()
    # 等待文件夹内容加载
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'file-list')))  # 替换为文件列表的定位方式

# 获取文件夹的最新更新时间
def get_latest_mod_time(driver):
    # 解析文件列表
    file_list = driver.find_element(By.CLASS_NAME, 'file-list')  # 替换为文件列表的定位方式
    files = file_list.find_elements(By.TAG_NAME, 'tr')  # 假设每行是一个文件
    latest_mod_time = None
    for file in files:
        cells = file.find_elements(By.TAG_NAME, 'td')
        if len(cells) >= 2:  # 假设第二列是更新时间
            file_name = cells[0].text
            mod_time = cells[1].text
            if not latest_mod_time or mod_time > latest_mod_time:
                latest_mod_time = mod_time
                latest_file = file_name
    return latest_file, latest_mod_time

# 检查文件夹是否有更新
def check_for_updates(driver, folder_name):
    # 进入文件夹
    enter_folder(driver, folder_name)
    # 获取最新更新时间
    result = get_latest_mod_time(driver)
    if not result:
        return

    latest_file, latest_mod_time = result

    # 检查是否有更新
    if not os.path.exists(LAST_MOD_TIME_FILE):
        with open(LAST_MOD_TIME_FILE, 'w') as f:
            f.write(latest_mod_time)
        print(f"初始检查：最新文件是 {latest_file}，更新时间：{latest_mod_time}")
    else:
        with open(LAST_MOD_TIME_FILE, 'r') as f:
            last_mod_time = f.read().strip()

        if latest_mod_time != last_mod_time:
            print(f"检测到更新：{latest_file}，更新时间：{latest_mod_time}")
            # 更新最后修改时间
            with open(LAST_MOD_TIME_FILE, 'w') as f:
                f.write(latest_mod_time)
        else:
            print("自上次检查以来没有更新")

if __name__ == '__main__':
    driver = init_driver()
    try:
        # 加载Cookie
        load_cookie(driver)

        while True:
            check_for_updates(driver, FOLDER_NAME)
            time.sleep(60 * 5)  # 每5分钟检查一次
    finally:
        driver.quit()  # 关闭浏览器