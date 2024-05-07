# -*- coding: utf-8 -*-
# @Time    : 5/7/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : data.py

from typing import List, Any, Optional


def chunk(data: List[Any], n_chunks: Optional[int] = None, chunk_size: Optional[int] = None) -> List[List[Any]]:
    if n_chunks is not None and chunk_size is not None:
        raise ValueError("Only one of n_chunks or chunk_size can be set")
    elif n_chunks is None and chunk_size is None:
        raise ValueError("One of n_chunks or chunk_size must be set")

    if n_chunks:
        chunk_size = len(data) // n_chunks

    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
