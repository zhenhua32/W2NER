import logging
import pickle
import time
from collections import defaultdict, deque


def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    """
    将索引和实体类型转换成文本格式, 结构形如 0-1-2-#-id, -#- 前面是索引列表, 后面是标签对应的 id
    """
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    """
    从文本格式解析为索引和实体类型
    """
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def decode(outputs, entities, length):
    class Node:
        def __init__(self):
            # 保存的是 tail 对应的 type_id
            self.THW = []  # [(tail, type)]
            # 保存的是 head, tail 对应的 next_index
            self.NNW = defaultdict(set)  # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()
    """
    标签是 None、NNW和THW-*，因此共有实体类别数+2个F1值
    """
    for instance, ent_set, l in zip(outputs, entities, length):
        # instance 是二维结构, 是预测出来的标签
        # 处理单条数据, instance 是当前的预测结果, ent_set 是实体集合, l 是序列长度
        predicts = []
        nodes = [Node() for _ in range(l)]  # 是列组成的, 第N个节点表示序列的顺序
        # 反向遍历, 从 l-1 到 0, 从序列的尾部开始
        for cur in reversed(range(l)):
            heads = []
            # 从 0 到 cur, cur 一开始是 l-1, 依次减小
            for pre in range(cur + 1):
                # THW
                # > 1 说明是 THW-* 系列的标签, 也就是实体标签
                # 这里的 [cur, pre] 遍历就是从最底部的行开始, 遍历到最上边
                if instance[cur, pre] > 1:
                    # cur 是行, pre 是列. 第 pre 列中加入一个 cur 行的 实体标签 type_id
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    # pre 是列, 也是开头, 一个列中的某个行有 THW 就是结尾
                    heads.append(pre)
                # NNW
                # == 1 说明是 NNW 标签, 注意这里是 instance[pre, cur], 和上面添加 THW 时是反向的.
                # 左下角是 THW 区域, 右上角是 NNW 区域
                # pre < cur 说明相等的不要, 因为这是标记 NNW, 需要下一个相邻的词, 所以不可能是相同的
                # 这里的 [pre, cur] 遍历就是从最右边的列开始, 遍历到最左边
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        # 在 (head, cur) 添加当前的索引位置 cur
                        nodes[pre].NNW[(head, cur)].add(cur)
                    # post nodes
                    # 注意这里遍历的是 nodes[cur] 这列的数据
                    for head, tail in nodes[cur].NNW.keys():
                        # 当前的位置在 (pre, cur) 是 (head, tail) 的内部, 所以添加 cur
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head, tail)].add(cur)

            # entity
            # 上面遍历完了 `for pre in range(cur + 1):`, 就可以开始找出这一行对应的实体了.
            # 注意这里是 nodes[cur], 上面添加的都是 nodes[pre]
            for tail, type_id in nodes[cur].THW:
                # 这种情况就是单个词的实体
                if cur == tail:
                    # 所以, predicts 中每一项都是 (index, type_id) 组成的
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                # 这个循环就是获取 cur 到 tail 的词
                while len(q) > 0:
                    chains = q.pop()
                    # chains 的最后一个值, 对应的 NNW
                    # chains 会一点点变大, 从 cur 开始, 直到 tail, 就是所有的相邻的词
                    for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])

        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)  # 真实实体的数量
        ent_p += len(predicts)  # 预测到的实体数量
        ent_c += len(predicts.intersection(ent_set))  # 真实实体和预测实体的交集
    return ent_c, ent_p, ent_r, decode_entities


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r
