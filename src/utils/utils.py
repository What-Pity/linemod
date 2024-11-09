import numpy as np

def binary_search_position(n_list, n):
    left, right = 0, len(n_list) - 1
    while left <= right:
        mid = (left + right) // 2
        if n_list[mid] < n:
            left = mid + 1
        else:
            right = mid - 1
    return left


if __name__ == '__main__':
    # 定义变量
    n = 19.1
    n_list = [1, 2, 3, 4, 5, 6]
    # 调用函数
    position = binary_search_position(n_list, n)
    print(f"数字 {n} 应该插入的位置是: {position}")