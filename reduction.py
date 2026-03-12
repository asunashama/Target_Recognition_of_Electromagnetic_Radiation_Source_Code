import numpy as np
from typing import Tuple

def pca_reduction(X: np.ndarray, retain_ratio: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    主成分分析降维
    :param X: 输入特征矩阵 (n_samples, n_features)
    :param retain_ratio: 保留的信息量比例 (默认0.95)
    :return: 降维后特征矩阵 Y (n_samples, d)，变换矩阵 W (d, n_features)
    """
    # 获取样本数和特征数
    n_samples, n_features = X.shape
    
    # 数据中心化（减去均值）
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # (1) 计算样本协方差矩阵 R = XX^T
    # 注意：公式(8.28)中 R = XX^T，但通常PCA使用的是 X^T X 或 (1/(n-1)) X^T X
    # 为了与特征值分解一致，这里使用 X @ X.T 或根据样本数调整
    if n_samples < n_features:
        # 当样本数小于特征数时，使用小技巧计算协方差矩阵
        # 计算 Gram 矩阵 R = X_centered @ X_centered.T (n_samples × n_samples)
        R = X_centered @ X_centered.T
        R = R / (n_samples - 1)  # 无偏估计
        use_gram = True
    else:
        # 标准协方差矩阵 R = X_centered.T @ X_centered (n_features × n_features)
        R = X_centered.T @ X_centered
        R = R / (n_samples - 1)  # 无偏估计
        use_gram = False
    
    # (2) 对协方差矩阵进行特征值分解 R = U Λ U^T
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    
    # (3) 对特征值从大到小排序
    idx = np.argsort(eigenvalues)[::-1]  # 降序排列
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 处理使用Gram矩阵的情况
    if use_gram:
        # 当使用 Gram 矩阵时，需要转换回原始特征空间的特征向量
        # 关系: U_feat = X_centered.T @ U_gram / sqrt(eigenvalues)
        eigenvectors_feat = X_centered.T @ eigenvectors
        # 归一化特征向量
        for i in range(len(eigenvalues)):
            if eigenvalues[i] > 1e-10:
                eigenvectors_feat[:, i] = eigenvectors_feat[:, i] / np.sqrt(eigenvalues[i] * (n_samples - 1))
        eigenvectors = eigenvectors_feat
    
    # (4) 计算信息量并选择主成分数量
    # 计算累积信息量
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues)
    cumulative_ratio = cumulative_variance / total_variance
    
    # 找到满足信息量比例的最小 d
    d = np.searchsorted(cumulative_ratio, retain_ratio) + 1
    d = min(d, n_features)  # 确保不超过特征数
    
    print(f"PCA降维: 原始维度 {n_features} -> 保留维度 {d}, "
          f"信息量保留比例: {cumulative_ratio[d-1]:.4f}")
    
    # 选取前 d 个特征向量构成映射矩阵 W
    W = eigenvectors[:, :d].T  # (d, n_features)
    
    # (5) 将 X 通过映射矩阵投影到特征空间 Y = W^T X
    # 注意：需要先中心化数据
    Y = (W @ X_centered.T).T  # (n_samples, d)
    
    return Y, W

import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
import warnings

def udp_reduction(X: np.ndarray, n_components: int, k_neighbors: int = 5) -> np.ndarray:
    """
    无监督鉴别投影降维
    :param X: 输入特征矩阵 (n_samples, n_features)
    :param n_components: 输出维数
    :param k_neighbors: 近邻数
    :return: 降维后特征矩阵 Y (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    
    if n_samples < 2:
        warnings.warn("样本数不足，返回原始数据")
        return X
    
    if n_components > n_features:
        warnings.warn(f"输出维数不能超过特征数，调整为 {n_features}")
        n_components = n_features
    
    print(f"UDP降维: 样本数={n_samples}, 特征数={n_features}, 近邻数={k_neighbors}")
    
    # 数据中心化
    X_centered = X - np.mean(X, axis=0)
    
    # (1) 构建邻接矩阵 H
    print("步骤1: 构建邻接矩阵...")
    
    # 使用KNN找到每个样本的k个近邻
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # 构建邻接矩阵 H (N × N)
    H = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        # indices[i, 1:] 排除自身（第一个索引是自身）
        for j in indices[i, 1:]:  # 跳过自身
            H[i, j] = 1
            H[j, i] = 1  # 对称化
    
    # 确保对角线为0
    np.fill_diagonal(H, 0)
    
    # 统计近邻关系
    n_edges = np.sum(H) / 2  # 无向图边数
    print(f"  构建了近邻图: {int(n_edges)} 条边")
    
    # (2) 计算局部散度矩阵 S_L (公式8.34)
    print("步骤2: 计算局部散度矩阵 S_L...")
    
    S_L = np.zeros((n_features, n_features))
    normalization = 2 * n_samples * n_samples  # 归一化因子 2·NN
    
    # 方法1: 直接计算（内存消耗大，但精确）
    # 对于小数据集可以使用
    if n_samples < 1000:  # 样本数小于1000时直接计算
        for i in range(n_samples):
            for j in range(n_samples):
                if H[i, j] > 0:
                    diff = X_centered[i] - X_centered[j]
                    diff = diff.reshape(-1, 1)
                    S_L += diff @ diff.T
    else:
        # 方法2: 使用稀疏计算（内存友好）
        # 找出所有非零的H[i,j]
        rows, cols = np.where(H > 0)
        for i, j in zip(rows, cols):
            if i < j:  # 只计算一次，利用对称性
                diff = X_centered[i] - X_centered[j]
                diff = diff.reshape(-1, 1)
                S_L += 2 * (diff @ diff.T)  # 乘以2因为H是对称的
    
    S_L = S_L / normalization
    
    # (3) 计算非局部散度矩阵 S_N (公式8.35)
    print("步骤3: 计算非局部散度矩阵 S_N...")
    
    S_N = np.zeros((n_features, n_features))
    
    # 方法: 计算所有点对的总和，然后减去局部部分
    # 计算所有点对的总和
    total_sum = np.zeros((n_features, n_features))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            diff = X_centered[i] - X_centered[j]
            diff = diff.reshape(-1, 1)
            total_sum += 2 * (diff @ diff.T)  # 乘以2因为对称
    
    # 减去局部部分得到非局部部分
    S_N = total_sum - S_L * normalization
    S_N = S_N / normalization
    
    # 验证 S_L + S_N 应该等于总散度矩阵
    total_scatter = np.zeros((n_features, n_features))
    for i in range(n_samples):
        total_scatter += X_centered[i].reshape(-1, 1) @ X_centered[i].reshape(1, -1)
    total_scatter = total_scatter * 2 / normalization
    
    # (4) 求解广义特征值问题 S_N W = λ S_L W (公式8.36)
    print("步骤4: 求解广义特征值问题...")
    
    # 添加正则化项处理 S_L 奇异的问题
    reg_term = 1e-6 * np.trace(S_L) / n_features * np.eye(n_features)
    S_L_reg = S_L + reg_term
    
    try:
        # 使用scipy求解广义特征值问题
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(S_N, S_L_reg)
        
        # 取实部
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # 按特征值降序排列（最大化 J(W) = W^T S_N W / W^T S_L W）
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 取前 n_components 个特征向量构成投影矩阵
        W = eigenvectors[:, :n_components]
        
        # 检查特征值是否有效
        valid_components = np.sum(eigenvalues[:n_components] > 1e-10)
        if valid_components < n_components:
            print(f"  警告: 只有 {valid_components} 个有效成分")
        
    except ImportError:
        # 如果没有scipy，使用numpy的eig求解（转换为普通特征值问题）
        print("  使用numpy求解特征值问题...")
        try:
            S_L_inv = np.linalg.pinv(S_L_reg)  # 使用伪逆
            A = S_L_inv @ S_N
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            W = eigenvectors[:, :n_components]
            
        except np.linalg.LinAlgError:
            warnings.warn("矩阵奇异，使用PCA作为后备")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            pca.fit(X)
            W = pca.components_.T
    
    # (5) 投影得到降维后的数据 Y = W^T X
    print("步骤5: 数据投影降维...")
    Y = X_centered @ W
    
    # 解释投影方向的信息量（近似）
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)
    print(f"  前{n_components}个方向累积信息量: {cumulative_variance[-1]:.4f}")
    
    return Y