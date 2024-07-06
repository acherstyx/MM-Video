# -*- coding: utf-8 -*-
# @Time    : 7/6/24
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : test_debug.py
import json
from pathlib import Path
from typing import Any

from pytest import mark, param

from mm_video.utils.common.debug import dump_return


@mark.parametrize(
    "test_data",
    [
        param({"test": "return_value"}, id="dict"),
        param([1, 2, 3], id="list"),
    ]
)
def test_dump_return(test_data: Any, dump_file: str = "test_output/dump_return.json"):
    def func():
        return test_data

    wrapped_func = dump_return(dump_file, keep_return=False)(func)
    assert wrapped_func() is None

    wrapped_func = dump_return(dump_file, keep_return=True)(func)
    assert wrapped_func() == test_data

    with open(dump_file) as f:
        dump_data = json.load(f)
    assert dump_data == test_data

    Path(dump_file).unlink(missing_ok=True)


@mark.parametrize(
    "test_data",
    [
        param({"test": "return_value"}, id="dict"),
        param([1, 2, 3], id="list"),
    ]
)
def test_dump_return_on_method(test_data: Any):
    class MyClass:
        @dump_return("test_output/method_return.json", keep_return=True)
        def method(self, test_data: Any):
            return test_data

    obj = MyClass()

    # Test with keep_return=True
    dump_file = "test_output/method_return.json"
    result = obj.method(test_data)
    assert result == test_data

    with open(dump_file) as f:
        dump_data = json.load(f)
    assert dump_data == test_data

    # Clean up
    Path(dump_file).unlink(missing_ok=True)

    # Test with keep_return=False
    class MyClassNoReturn:
        @dump_return(dump_file, keep_return=False)
        def method(self, test_data: Any):
            return test_data

    obj_no_return = MyClassNoReturn()
    result = obj_no_return.method(test_data)
    assert result is None

    with open(dump_file) as f:
        dump_data = json.load(f)
    assert dump_data == test_data

    # Clean up
    Path(dump_file).unlink(missing_ok=True)
