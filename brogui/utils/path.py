# -*- coding: utf-8 -*-
"""
  Copyright 2022 Mitchell Isaac Parker

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import os

util_str = "util"
data_str = "data"

def get_dir_name(dir_path):

    if "/" in dir_path:
        dir_name = dir_path.rsplit("/", 1)[0]
    else:
        dir_name = os.getcwd()

    return dir_name

def get_dir_path(dir_str=None, dir_path=None):

    if dir_path is None:
        dir_path = os.getcwd()

    if dir_str is not None:
        dir_path += f"/{dir_str}"

    return dir_path


def get_file_path(file_name, dir_str=None, dir_path=None, pre_str=True):

    file_path = get_dir_path(dir_str=dir_str, dir_path=dir_path)
    file_path += "/"
    if pre_str and dir_str != None:
        file_path += dir_str
        file_path += "_"
    file_path += file_name

    return file_path