import datetime
import json
import ntpath
import os


def extract_crossname(path):
    # 使用 ntpath.basename 来处理 Windows 路径
    filename = ntpath.basename(path)
    # 分割文件名和扩展名
    name_parts = filename.split('.')
    # 返回第一个部分（基本文件名）
    return name_parts[0]


def create_file_if_not_exists(filename):
    # 获取文件所在的目录路径
    directory = os.path.dirname(filename)
    # 如果目录不存在，创建目录
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    # 如果文件不存在，创建文件
    if not os.path.exists(filename):
        try:
            with open(filename, 'w') as f:
                pass  # 创建一个空文件
            print(f"Created file: {filename}")
        except IOError as e:
            print(f"Error creating file {filename}: {e}")
            return False
    else:
        print(f"File already exists: {filename}")
    return True


def add_directory_if_missing(path, directory="./"):
    # 规范化路径分隔符
    path = os.path.normpath(path)
    # 分割路径
    path_parts = os.path.split(path)
    # 检查是否已经包含目录
    if path_parts[0]:
        return path
    else:
        # 如果没有目录，添加指定的目录
        return os.path.join(directory, path_parts[1])


def write_eval_result(mean, std, filename="eval_result.txt"):
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 将时间和变量组合成一行
    line = f"{current_time}, {mean}, {std}\n"

    create_file_if_not_exists(filename)
    # 以写入模式打开文件并写入
    with open(filename, "a") as file:
        file.write(line)
    print(f"Data written to {filename}")


def write_predict_result(data, filename='predict_results.json', print_to_console=False):
    create_file_if_not_exists(filename)

    if print_to_console:
        print(data)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)