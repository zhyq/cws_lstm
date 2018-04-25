import numpy as np
import pandas as pd
xrange = range
class Viterbi():
    def __init__(self,labels):
        # 利用 labels（即状态序列）来统计转移概率
        # 因为状态数比较少，这里用 dict={'I_tI_{t+1}'：p} 来实现
        # A统计状态转移的频数
        A = {
              'sb':0,
              'ss':0,
              'be':0,
              'bm':0,
              'me':0,
              'mm':0,
              'eb':0,
              'es':0
             }

        # zy 表示转移概率矩阵
        zy = dict()
        for label in labels:
            for t in xrange(len(label) - 1):
                key = label[t] + label[t+1]
                A[key] += 1.0

        zy['sb'] = A['sb'] / (A['sb'] + A['ss'])
        zy['ss'] = 1.0 - zy['sb']
        zy['be'] = A['be'] / (A['be'] + A['bm'])
        zy['bm'] = 1.0 - zy['be']
        zy['me'] = A['me'] / (A['me'] + A['mm'])
        zy['mm'] = 1.0 - zy['me']
        zy['eb'] = A['eb'] / (A['eb'] + A['es'])
        zy['es'] = 1.0 - zy['eb']
        keys = sorted(zy.keys())
        print ("the transition probability: ")
        for key in keys:
            print( key, zy[key])
        zy = {i:np.log(zy[i]) for i in zy.keys()}
        self.A = A
        self.keys = keys
        self.zy = zy
    
    def viterbi(self,nodes):
        """
        维特比译码：除了第一层以外，每一层有4个节点。
        计算当前层（第一层不需要计算）四个节点的最短路径：
         对于本层的每一个节点，计算出路径来自上一层的各个节点的新的路径长度（概率）。保留最大值（最短路径）。
         上一层每个节点的路径保存在 paths 中。计算本层的时候，先用paths_ 暂存，然后把本层的最大路径保存到 paths 中。
         paths 采用字典的形式保存（路径：路径长度）。
         一直计算到最后一层，得到四条路径，将长度最短（概率值最大的路径返回）
        """


        paths = {'b': nodes[0]['b'], 's':nodes[0]['s']} # 第一层，只有两个节点
        for layer in xrange(1, len(nodes)):  # 后面的每一层
            paths_ = paths.copy()  # 先保存上一层的路径
            # node_now 为本层节点， node_last 为上层节点
            paths = {}  # 清空 path
            for node_now in nodes[layer].keys():
                # 对于本层的每个节点，找出最短路径
                sub_paths = {}
                # 上一层的每个节点到本层节点的连接
                for path_last in paths_.keys():
                    if path_last[-1] + node_now in self.zy.keys(): # 若转移概率不为 0
                        sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + self.zy[path_last[-1] + node_now]
                # 最短路径,即概率最大的那个
                sr_subpaths = pd.Series(sub_paths)
                sr_subpaths = sr_subpaths.sort_values()  # 升序排序
                node_subpath = sr_subpaths.index[-1]  # 最短路径
                node_value = sr_subpaths[-1]   # 最短路径对应的值
                # 把 node_now 的最短路径添加到 paths 中
                paths[node_subpath] = node_value
        # 所有层求完后，找出最后一层中各个节点的路径最短的路径
        sr_paths = pd.Series(paths)
        sr_paths = sr_paths.sort_values()  # 按照升序排序
        return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）
