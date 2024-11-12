import numpy as np

def check_npy_file(file_path):
    # 加载.npy文件
    data = np.load(file_path)

    # 打印数组的形状
    print("Array shape:", data.shape)

    # 打印数组的数据类型
    print("Data type:", data.dtype)

    # 打印数组的一些统计信息
    print("Minimum value:", np.min(data))
    print("Maximum value:", np.max(data))
    print("Mean value:", np.mean(data))
    print("Standard deviation:", np.std(data))

    # 检查数组是否包含NaN值
    if np.isnan(data).any():
        print("Warning: The array contains NaN values.")
    else:
        print("The array does not contain NaN values.")

    # 检查数组是否包含无穷大值
    if np.isinf(data).any():
        print("Warning: The array contains infinite values.")
    else:
        print("The array does not contain infinite values.")

    # 可以添加更多的检查，如特定值的计数等
    # print(data)

# 使用示例
file_path = r'data\test_label.npy'  # 替换为你的 .npy文件路径
check_npy_file(file_path)