import heapq


def operon_topk(inds, k):
    """
    TopK算法
    :param nums: 数据集
    :param k: 前K个元素
    :return: 前K个元素
    """
    heap = []
    for ind in inds:
        if len(heap) < k:
            heapq.heappush(heap, num)
        else:
            if num > heap[0]:
                heapq.heappushpop(heap, num)
    return heap