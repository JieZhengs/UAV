import pickle

import numpy as np

import pickle
import numpy as np

def check_pkl_file_and_print(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f"文件 {file_path} 是有效的 .pkl 文件，内容已加载。")
            # print(sample)
            print("文件内容如下：")
            print(len(data[0]))
            print(type(data))  # 打印文件内容

            # 检查数据是否为 NumPy 数组或包含 NumPy 数组的列表
            if isinstance(data, np.ndarray):
                print("数据维度：", data.shape)
            elif isinstance(data, list) and all(isinstance(item, np.ndarray) for item in data):
                print("数据是包含多个 NumPy 数组的列表。")
                for idx, item in enumerate(data):
                    print(f"数组 {idx} 的维度：{item.shape}")
            else:
                print("数据不是 NumPy 数组，无法获取维度。")

            return True
    except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
        print(f"文件 {file_path} 不是有效的 .pkl 文件或文件不存在。错误信息：{e}")
        return False

# 使用示例
file_path = './data\\uav\\xsub1\\v1_label_data_append.pkl'  # 替换为你的 .pkl 文件路径
check_pkl_file_and_print(file_path)