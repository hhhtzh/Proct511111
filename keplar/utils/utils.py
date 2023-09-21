import heapq
import sys


# def operon_topk(inds, k):
#     """
#     TopK算法
#     :param nums: 数据集
#     :param k: 前K个元素
#     :return: 前K个元素
#     """
#     heap = []
#     for ind in inds:
#         if len(heap) < k:
#             heapq.heappush(heap, num)
#         else:
#             if num > heap[0]:
#                 heapq.heappushpop(heap, num)
#     return heap

def heapify(ind_arr, n, i):
    smallest = i
    left = 2 * i + 1
    right = 2 * i + 2

    # 找到左子节点和右子节点中的最小值索引
    if left < n and ind_arr[i].GetFitness(0) > ind_arr[left].GetFitness(0):
        smallest = left

    if right < n and ind_arr[smallest].GetFitness(0) > ind_arr[right].GetFitness(0):
        smallest = right

    # 如果最小值不是当前节点，交换它们，并递归地对受影响的子树进行堆化
    if smallest != i:
        ind_arr[i], ind_arr[smallest] = ind_arr[smallest], ind_arr[i]
        heapify(ind_arr, n, smallest)


def build_min_heap(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)


def find_k_smallest_elements(ind_arr, k):
    if k <= 0:
        return []

    # 构建最小堆
    min_heap = ind_arr[:k]
    build_min_heap(min_heap)

    # 继续遍历剩余元素，如果比堆顶小，则替换堆顶并重新堆化
    for i in range(k, len(ind_arr)):
        if ind_arr[i].GetFitness(0) < min_heap[0].GetFitness(0):
            min_heap[0] = ind_arr[i]
            heapify(min_heap, k, 0)

    return min_heap


# 示例
# if __name__ == "__main__":
#     arr = [5, 9, 2, 8, 3, 7, 1, 6, 4]
#     k = 3
#     result = find_k_smallest_elements(ind_arr, k)
#     print("前", k, "个最小的数:", result)
# def heapify(ind_arr, n, i):
#     largest = i  # 假设最大值在当前节点
#     left = 2 * i + 1
#     right = 2 * i + 2
#
#     # 找到左子节点和右子节点中的最大值索引
#     if left < n and ind_arr[i].GetFitness(0) < ind_arr[left].GetFitness(0):
#         largest = left
#
#     if right < n and ind_arr[largest].GetFitness(0) < ind_arr[right].GetFitness(0):
#         largest = right
#
#     # 如果最大值不是当前节点，交换它们，并递归地对受影响的子树进行堆化
#     if largest != i:
#         ind_arr[i], ind_arr[largest] = ind_arr[largest], ind_arr[i]
#         heapify(ind_arr, n, largest)
#
#
# def operon_heap_sort(ind_arr):
#     n = len(ind_arr)
#
#     # 构建最大堆，从最后一个非叶子节点开始向上堆化
#     for i in range(n // 2 - 1, -1, -1):
#         print(i)
#         heapify(ind_arr, n, i)
#
#     # 从堆中一个个取出元素，进行排序
#     for i in range(n - 1, 0, -1):
#         print(i)
#         ind_arr[i], ind_arr[0] = ind_arr[0], ind_arr[i]  # 将最大值（根节点）放到末尾
#         heapify(ind_arr, i, 0)  # 重新堆化剩余的元素
#
#
# # # 示例
# # if __name__ == "__main__":
# #     arr = [12, 11, 13, 5, 6, 7]
# #     print("原始数组:", arr)
# #     operon_heap_sort(arr)
# #     print("排序后的数组:", arr)
#
# # # 原始数组: [12, 11, 13, 5, 6, 7]
# # # 排序后的数组: [5, 6, 7, 11, 12, 13]
# for i in range(4, 0, -1):
#     print(i)
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
