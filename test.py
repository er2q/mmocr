# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : test.py
# @datetime: 2022/8/24 10:33
# @software: PyCharm

"""
文件说明：
    
"""
items = [1, 2, 3, 4]
for idx, item in enumerate(items):
    items.remove(item)

print(items)
