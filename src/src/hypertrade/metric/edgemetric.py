"""高阶网络的超边指标

"""

from . import np

def get_avg_degree_by_edge_size(H):
    """计算每个超边大小 m 对应的节点平均度数。

    Parameters
    ----------
    H: hypernetx.Hypergraph 
        超图对象

    Returns
    -------
    dict[int, float]
        键为超边大小 m (int)，值为节点平均度数 (float)
    """
    # 1. 获取关联矩阵及其行列索引
    M, row_idx, col_idx = H.incidence_matrix(index=True)

    # 2. 计算每条超边的大小 (即关联矩阵每列的元素之和)
    # 使用 np.array().flatten() 将 scipy 矩阵对象转换为一维 numpy 数组
    edge_sizes = np.array(M.sum(axis=0)).flatten()

    # 3. 获取超图中所有出现过的超边大小 m (去重)
    unique_sizes = np.unique(edge_sizes)

    # 4. 初始化结果字典
    result_dict = {}

    # 5. 遍历每个超边大小 m
    for m in unique_sizes:
        # 找到大小为 m 的超边对应的列索引掩码 (Mask)
        mask = (edge_sizes == m)

        # 提取只包含大小为 m 的超边的子关联矩阵
        M_sub = M[:, mask]

        # 计算每个节点在大小为 m 的超边中的度数 (即子矩阵每行的元素之和)
        node_degrees = np.array(M_sub.sum(axis=1)).flatten()

        # 计算平均值
        # 默认情况：除以超图中的【总节点数】（包含在该 m 下度数为 0 的节点）
        avg_degree = np.mean(node_degrees)

        # ==========================================================
        # 【注意】关于“平均值”的定义：
        # 如果你只想计算【实际参与了大小为 m 的超边】的节点的平均度数
        # (即排除度数为 0 的节点)，请取消下方注释，并注释掉上面的 avg_degree 计算：
        #
        # non_zero_degrees = node_degrees[node_degrees > 0]
        # avg_degree = np.mean(non_zero_degrees) if len(non_zero_degrees) > 0 else 0.0
        # ==========================================================

        # 将结果存入字典，确保键为 int，值为 float
        result_dict[int(m)] = float(avg_degree)

    return result_dict
