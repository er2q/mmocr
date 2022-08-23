# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : 4.读取TXT.py
# @datetime: 2022/8/20 14:45
# @software: PyCharm

"""
文件说明：
    
"""
with open("./keys.txt", "r") as f:  # 打开文件
    data = f.read().replace('\n', '')  # 读取文件
    # print(data)

str = data
same = ''
diff = ''
for i in range(len(str)):
    if str.count(str[i]) > 1:
        same += str[i]
    else:
        diff += str[i]
print('重复的元素有：%s' % same)
print('不重复的元素有：%s' % diff)
