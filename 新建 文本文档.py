# encoding: utf-8
'''
  @author 李华鑫
  @create 2020-10-08 8:27
  Mycsdn：https://buwenbuhuo.blog.csdn.net/
  @contact: 459804692@qq.com
  @software: Pycharm
  @file: 作业：网易云音乐.py
  @Version：1.0
  
'''
"""
华语男歌手： https://music.163.com/discover/artist/cat?id=1001
华语女歌手： https://music.163.com/discover/artist/cat?id=1002
"""
import requests
import random
import csv
import time
from lxml import etree

# num = [1001,1002,1003,2001,2002,2003,6001,6002,6003,7001,7002,7003,4001,4002,4003]
base_url = "https://music.163.com/"
# start_url = "https://music.163.com/discover/artist/cat?id=1001"
start_url = "https://music.163.com/discover/artist/"
headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36',
}

items = []

def parse_url(url):
    """解析url，得到响应内容"""
    time.sleep(random.random())
    response = requests.get(url=url,headers=headers)
    return response.content.decode("utf-8")


def parse_html(html):
    """使用xpath解析html,返回xpath对象"""
    etree_obj = etree.HTML(html)
    return etree_obj


def get_type_url():
    """获取所有的歌手类型"""
    types = []

    html = parse_url(start_url)
    etree_obj =parse_html(html)
    type_name_list = etree_obj.xpath('//a[@class="cat-flag"]/text()')
    # print(type_name_list)
    type_url_list = etree_obj.xpath('//a[@class="cat-flag"]/@href')
    data_zip = zip(type_name_list[1:],type_url_list[1:])

    for data in data_zip:
        type = {}
        type["name"] = data[0]
        type["url"] = data[1]
        types.append(type)

    return types

def get_data(url, type_name):
    """爬歌手数据"""
    item = {
        "type": type_name,
        "name": "",
        "url": ""
    }

    html = parse_url(url)
    etree_obj = parse_html(html)
    artist_name_list = etree_obj.xpath('//a[@class="nm nm-icn f-thide s-fc0"]/text()')
    artist_url_list = etree_obj.xpath('//a[@class="nm nm-icn f-thide s-fc0"]/@href')

    data_zip = zip(artist_name_list, artist_url_list)
    for data in data_zip:
        item["name"] = data[0]
        item["url"] = base_url + data[1][1:]
        items.append(item)

def save():
    """将数据保存到csv中"""
    with open("./wangyinyun.csv", "a", encoding="utf-8") as file:
        writer = csv.writer(file)
        for item in items:
            writer.writerow(item.values())



def start():
    """开始爬虫"""
    types = get_type_url()
    # print(types)
    for type in types:
        # url = base_url+type["url"]
        # url还不够完整
        # print(url)
        for i in range(65,91):
            url = "{}{}&initial={}".format(base_url,type["url"],i)
            print(url)
            get_data(url, type["name"])
    save()
            # exit()

if __name__ == '__main__':
    start()

"""测试代码"""
# start_url = "https://music.163.com/discover/artist/cat?id=1001&initial=65"   a _ z
# response = requests.get(url=base_url,headers=headers)
# # print(response.content.decode("utf-8"))
# html = response.content.decode("utf-8")
# print(html)
# etree_obj = etree.HTML(html)
# # 只有华语男歌手
# # ret = etree_obj.xpath('//a[@class="cat-flag z-slt"]/text()')
# # 所有歌手
# ret = etree_obj.xpath('//a[@class="cat-flag"]/text()')
# print(ret)
# print(len(ret))
#
# # 链接
# ret = etree_obj.xpath('//a[@class="cat-flag"]/@href')
# print(ret)

"""
<li><a href="/discover/artist/cat?id=1001"
class="cat-flag z-slt"
data-cat="1001">华语男歌手</a>
</li>

https://music.163.com/discover/artist/cat?id=1001&initial=65
"""