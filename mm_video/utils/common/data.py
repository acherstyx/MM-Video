# -*- coding: utf-8 -*-
# @Time    : 5/7/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : data.py

from typing import List, Any, Optional


def chunk(data: List[Any], n_chunks: Optional[int] = None, chunk_size: Optional[int] = None) -> List[List[Any]]:
    if n_chunks is not None and chunk_size is not None:
        raise ValueError("Only one of n_chunks or chunk_size can be set")
    if n_chunks is None and chunk_size is None:
        raise ValueError("One of n_chunks or chunk_size must be set")

    if chunk_size:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    elif n_chunks:
        if n_chunks <= 0:
            raise ValueError("Number of chunks must be greater than 0")

        chunk_size = len(data) // n_chunks
        remainder = len(data) % n_chunks

        chunks = []
        start = 0
        for i in range(n_chunks):
            # Calculate the end index for the current chunk
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(data[start:end])
            start = end
        return chunks
    else:
        raise RuntimeError
