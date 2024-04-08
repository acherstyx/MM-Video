# -*- coding: utf-8 -*-
# @Time    : 2022/11/12 22:32
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : logging.py

import datetime


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
