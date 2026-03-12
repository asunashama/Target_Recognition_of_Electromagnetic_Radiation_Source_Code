import numpy as np
from typing import List, Tuple
from collections import Counter

def fisher_score_selection(X: np.ndarray, labels: np.ndarray, top_k: int) -> List[int]:
    """
    基于Fisher得分的特征选择
    :param X: 输入特征矩阵 (n_samples, n_features)
    :param labels: 样本标签 (n_samples,)
    :param top_k: 选择的特征个数
    :return: 选中的特征索引列表
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        raise ValueError("类别数必须大于等于2")
    
    if top_k > n_features:
        top_k = n_features
        print(f"top_k超过特征总数，调整为 {n_features}")
    
    # 计算各类别的样本数 m_k 和总样本数 M_t
    class_counts = Counter(labels)
    M_t = n_samples
    
    # 计算总体均值 μ(f_i) 对每个特征
    overall_means = np.mean(X, axis=0)  # shape: (n_features,)
    
    # 预计算每个类别的样本索引和统计量
    class_indices = {}
    class_means = np.zeros((n_classes, n_features))
    class_samples = {}
    
    for idx, label in enumerate(unique_labels):
        class_mask = (labels == label)
        class_indices[label] = np.where(class_mask)[0]
        class_samples[label] = X[class_mask]
        class_means[idx] = np.mean(class_samples[label], axis=0)
    
    # 计算每个特征的Fisher鉴别率
    fisher_scores = np.zeros(n_features)
    
    for i in range(n_features):
        # 计算类间离散度 S_b(f_i) (公式8.38)
        S_b = 0.0
        for idx, label in enumerate(unique_labels):
            m_k = class_counts[label]  # 第k类的样本个数
            mu_k = class_means[idx, i]  # 第k类第i维特征的均值
            mu = overall_means[i]  # 全部样本第i维特征的均值
            
            # 累加贡献: m_k * (μ_k - μ)^2 / M_t
            S_b += m_k * (mu_k - mu) ** 2 / M_t
        
        # 计算类内离散度 S_w(f_i) (公式8.39)
        S_w = 0.0
        for label in unique_labels:
            class_data = class_samples[label][:, i]  # 第k类第i维特征的所有值
            mu_k = np.mean(class_data)  # 第k类第i维特征的均值
            
            # 累加该类内所有样本的离差平方和
            S_w += np.sum((class_data - mu_k) ** 2)
        
        # 计算Fisher鉴别率 (公式8.37)
        if S_w > 1e-10:  # 避免除零
            fisher_scores[i] = S_b / S_w
        else:
            # 如果类内离散度为0，说明该类内所有样本在该特征上取值相同
            # 此时如果S_b > 0，则该特征具有完美区分度
            fisher_scores[i] = float('inf') if S_b > 0 else 0.0
    
    # 处理无穷大的情况
    inf_mask = np.isinf(fisher_scores)
    if np.any(inf_mask):
        print(f"发现 {np.sum(inf_mask)} 个特征的类内离散度为0")
    
    # 按Fisher鉴别率降序排序
    sorted_indices = np.argsort(fisher_scores)[::-1]  # 降序排列
    
    # 选择前top_k个特征索引
    selected_indices = sorted_indices[:top_k].tolist()
    
    # 输出选择结果统计
    print(f"特征选择完成: 从 {n_features} 维中选择 {top_k} 维")
    print(f"最大Fisher得分: {fisher_scores[selected_indices[0]]:.4f}")
    print(f"最小Fisher得分: {fisher_scores[selected_indices[-1]]:.4f}")
    
    # 可选：返回得分以供分析
    return selected_indices

import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
import warnings

def laplacian_score_selection(X: np.ndarray, top_k: int, sigma: float = 1.0, 
                              k_neighbors: int = 5) -> List[int]:
    """
    基于拉普拉斯得分的无监督特征选择
    :param X: 输入特征矩阵 (n_samples, n_features)
    :param top_k: 选择的特征个数
    :param sigma: 高斯核参数 t (默认1.0)
    :param k_neighbors: 近邻数 k (默认5)
    :return: 选中的特征索引列表
    """
    n_samples, n_features = X.shape
    
    if n_samples < 2:
        raise ValueError("样本数必须大于等于2")
    
    if top_k > n_features:
        top_k = n_features
        print(f"top_k超过特征总数，调整为 {n_features}")
    
    print(f"拉普拉斯特征选择: 样本数={n_samples}, 特征数={n_features}, 近邻数={k_neighbors}")
    
    # (1) 构造k近邻图 G
    print("步骤1: 构造k近邻图...")
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # (2) 计算相似矩阵 S (公式8.40)
    print("步骤2: 计算相似矩阵...")
    S = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j_idx, j in enumerate(indices[i, 1:]):  # 跳过自身
            if i < j:  # 只计算一次，利用对称性
                dist = distances[i, j_idx + 1]  # 对应距离
                # 使用高斯核计算相似度
                weight = np.exp(-dist / sigma)  # 公式8.40中使用exp(-||x_i - x_j||/t)
                S[i, j] = weight
                S[j, i] = weight  # 对称矩阵
    
    # 统计图信息
    n_edges = np.sum(S > 0) / 2
    print(f"  构建了相似图: {int(n_edges)} 条边")
    
    # (3) 构造对角矩阵 D (公式8.42)
    print("步骤3: 构造拉普拉斯矩阵...")
    D_diag = np.sum(S, axis=1)  # 每行和作为对角元
    D = np.diag(D_diag)
    
    # 构造拉普拉斯矩阵 L = D - S (公式8.41)
    L = D - S
    
    # (4) 对每一维特征计算拉普拉斯得分 (公式8.43)
    print("步骤4: 计算每维特征的拉普拉斯得分...")
    laplacian_scores = np.zeros(n_features)
    
    for r in range(n_features):
        # 获取第r维特征向量 f_r (公式8.44)
        f_r = X[:, r].reshape(-1, 1)  # N×1向量
        
        # 计算分子: f_r^T L f_r
        numerator = (f_r.T @ L @ f_r).item()
        
        # 计算分母: f_r^T D f_r = Var(f_r) * (某种归一化)
        # 公式8.43中分母为 Var(f_r)，这里使用 f_r^T D f_r 等价形式
        denominator = (f_r.T @ D @ f_r).item()
        
        # 避免除零
        if denominator > 1e-10:
            score = numerator / denominator
        else:
            # 如果分母为0，说明该维特征为常数
            score = float('inf') if numerator > 0 else 0
        
        laplacian_scores[r] = score
        
        # 可选：打印方差信息
        variance = np.var(f_r)
        if r < 5:  # 只打印前5个特征的信息作为示例
            print(f"    特征{r}: 得分={score:.4f}, 方差={variance:.4f}")
    
    # 另一种计算方式：使用公式中的方差形式验证
    # 计算所有特征的方差
    feature_variances = np.var(X, axis=0)
    
    # 计算每个特征的拉普拉斯得分的另一种形式
    laplacian_scores_alt = np.zeros(n_features)
    for r in range(n_features):
        f_r = X[:, r]
        # 计算 ∑_{ij} (f_{r,i} - f_{r,j})^2 S_{i,j}
        sum_sq_diff = 0
        for i in range(n_samples):
            for j in range(n_samples):
                if S[i, j] > 0:
                    sum_sq_diff += (f_r[i] - f_r[j])**2 * S[i, j]
        
        if feature_variances[r] > 1e-10:
            laplacian_scores_alt[r] = sum_sq_diff / (2 * feature_variances[r] * n_samples)
        else:
            laplacian_scores_alt[r] = float('inf')
    
    # 验证两种计算方法的一致性
    correlation = np.corrcoef(laplacian_scores, laplacian_scores_alt)[0, 1]
    print(f"  两种计算方法的相关性: {correlation:.4f}")
    
    # (5) 按照拉普拉斯得分排序（得分越低越重要）
    print("步骤5: 特征排序与选择...")
    
    # 得分越低表示特征越重要
    sorted_indices = np.argsort(laplacian_scores)
    
    # 选择前top_k个特征（得分最低的）
    selected_indices = sorted_indices[:top_k].tolist()
    
    # 输出选择结果统计
    print(f"\n特征选择完成: 从 {n_features} 维中选择 {top_k} 维")
    print(f"最小得分: {laplacian_scores[selected_indices[0]]:.4f}")
    print(f"最大得分(选中): {laplacian_scores[selected_indices[-1]]:.4f}")
    print(f"未选最小得分: {laplacian_scores[sorted_indices[top_k]]:.4f}")
    
    # 打印选中特征的统计信息
    print("\n选中特征详情:")
    for i, idx in enumerate(selected_indices[:min(10, top_k)]):
        print(f"  特征{idx}: 得分={laplacian_scores[idx]:.4f}, 方差={feature_variances[idx]:.4f}")
    
    return selected_indices

import numpy as np
from typing import List, Tuple
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from sklearn.preprocessing import StandardScaler
import warnings

def mcfs_selection(X: np.ndarray, n_clusters: int, top_k: int, sigma: float = 1.0,
                   gamma: float = 1.0, max_iter: int = 1000) -> List[int]:
    """
    基于多聚类特征选择（MCFS）的无监督特征选择
    :param X: 输入特征矩阵 (n_samples, n_features)
    :param n_clusters: 聚类数（低维流形维数）
    :param top_k: 选择的特征个数
    :param sigma: 高斯核参数 (默认1.0)
    :param gamma: L1正则化参数 (默认1.0)
    :param max_iter: 最大迭代次数
    :return: 选中的特征索引列表
    """
    n_samples, n_features = X.shape
    
    if n_samples < 2:
        raise ValueError("样本数必须大于等于2")
    
    if n_clusters >= n_samples:
        n_clusters = n_samples - 1
        print(f"n_clusters调整为 {n_clusters}")
    
    if top_k > n_features:
        top_k = n_features
        print(f"top_k超过特征总数，调整为 {n_features}")
    
    print(f"MCFS特征选择: 样本数={n_samples}, 特征数={n_features}, 聚类数={n_clusters}")
    
    # 数据中心化
    X_centered = X - np.mean(X, axis=0)
    
    # (1) 利用高斯核函数构造相似矩阵 S (公式8.45)
    print("步骤1: 构造相似矩阵...")
    
    # 计算距离矩阵（使用近似方法避免O(N^2)内存）
    # 对于大规模数据，可以使用近似最近邻
    use_approximation = n_samples > 5000
    
    if use_approximation:
        print("  使用近似方法处理大规模数据...")
        from sklearn.neighbors import NearestNeighbors
        
        # 使用KNN近似构建稀疏相似矩阵
        n_neighbors = min(20, n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # 构建稀疏相似矩阵
        from scipy.sparse import csr_matrix
        rows = []
        cols = []
        data = []
        
        for i in range(n_samples):
            for j_idx, j in enumerate(indices[i]):
                if i != j:  # 排除自身
                    dist = distances[i, j_idx]
                    weight = np.exp(-dist**2 / (2 * sigma**2))
                    rows.append(i)
                    cols.append(j)
                    data.append(weight)
        
        S = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        # 对称化
        S = (S + S.T) / 2
        
    else:
        # 小规模数据，直接计算完整相似矩阵
        S = np.zeros((n_samples, n_samples))
        
        # 批量计算距离
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # 计算欧氏距离
                dist = np.linalg.norm(X[i] - X[j])
                # 高斯核相似度 (公式8.45)
                weight = np.exp(-dist**2 / (2 * sigma**2))
                S[i, j] = weight
                S[j, i] = weight
        
        # 对角线设为零
        np.fill_diagonal(S, 0)
    
    print(f"  相似矩阵构建完成，非零元素: {S.nnz if use_approximation else n_samples**2}")
    
    # (2) 求取相似矩阵的前d个最小特征值对应的特征向量
    print(f"步骤2: 提取前 {n_clusters} 个最小特征值对应的特征向量...")
    
    try:
        if use_approximation:
            # 对于稀疏矩阵，使用稀疏特征值求解器
            # 求最小的n_clusters个特征值
            eigenvalues, eigenvectors = eigs(S, k=n_clusters, which='SM', 
                                            maxiter=max_iter, tol=1e-6)
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
        else:
            # 对于稠密矩阵，使用标准特征值求解
            eigenvalues, eigenvectors = eig(S)
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            
            # 按特征值升序排列（取最小的）
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # 取前n_clusters个最小的
            eigenvectors = eigenvectors[:, :n_clusters]
        
        print(f"  特征值范围: [{np.min(eigenvalues):.4f}, {np.max(eigenvalues):.4f}]")
        
    except Exception as e:
        print(f"  特征值分解失败: {e}")
        # 使用随机投影作为后备
        print("  使用随机投影作为后备方案")
        eigenvectors = np.random.randn(n_samples, n_clusters)
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    
    # (3) 对所有特征向量求解系数回归问题 (公式8.46)
    print("步骤3: 求解稀疏回归系数...")
    
    # 初始化权重矩阵
    weights = np.zeros((n_clusters, n_features))
    
    for k in range(n_clusters):
        print(f"  处理第 {k+1}/{n_clusters} 个特征向量...")
        
        # 获取第k个特征向量 y_k
        y_k = eigenvectors[:, k].reshape(-1, 1)
        
        # 使用L1正则化最小二乘（Lasso）求解
        # 目标: min ||y_k - X^T a_k||^2 + gamma * |a_k|
        
        # 方法1: 使用坐标下降法求解Lasso
        a_k = solve_lasso(X_centered, y_k, gamma, max_iter=100)
        
        # 存储系数
        weights[k, :] = a_k.flatten()
    
    # (4) 计算每个特征的权重 MCFS(j) (公式8.47)
    print("步骤4: 计算特征权重...")
    
    # 取每个特征在所有特征向量上的最大系数绝对值
    mcfs_scores = np.max(np.abs(weights), axis=0)
    
    # 归一化得分
    if np.max(mcfs_scores) > 0:
        mcfs_scores = mcfs_scores / np.max(mcfs_scores)
    
    # 按权重从大到小排序
    sorted_indices = np.argsort(mcfs_scores)[::-1]  # 降序排列
    
    # 选择前top_k个特征
    selected_indices = sorted_indices[:top_k].tolist()
    
    # 输出结果统计
    print(f"\n特征选择完成: 从 {n_features} 维中选择 {top_k} 维")
    print(f"最大权重: {mcfs_scores[selected_indices[0]]:.4f}")
    print(f"最小权重(选中): {mcfs_scores[selected_indices[-1]]:.4f}")
    
    # 打印前10个选中特征的信息
    print("\n选中特征详情:")
    for i, idx in enumerate(selected_indices[:min(10, top_k)]):
        print(f"  特征{idx}: 权重={mcfs_scores[idx]:.4f}")
    
    return selected_indices


def solve_lasso(X: np.ndarray, y: np.ndarray, gamma: float, 
                max_iter: int = 100, tol: float = 1e-4) -> np.ndarray:
    """
    使用坐标下降法求解Lasso问题
    min ||y - X^T a||^2 + gamma * |a|
    """
    n_samples, n_features = X.shape
    
    # 初始化系数
    a = np.zeros((n_features, 1))
    
    # 缓存X^T X和X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    
    # 坐标下降迭代
    for iteration in range(max_iter):
        a_old = a.copy()
        
        # 更新每个坐标
        for j in range(n_features):
            # 计算残差
            residual_j = Xty[j] - (XtX[j, :] @ a).item() + XtX[j, j] * a[j].item()
            
            # 软阈值操作
            if residual_j > gamma:
                a[j] = (residual_j - gamma) / XtX[j, j]
            elif residual_j < -gamma:
                a[j] = (residual_j + gamma) / XtX[j, j]
            else:
                a[j] = 0
        
        # 检查收敛
        if np.linalg.norm(a - a_old) < tol:
            break
    
    return a


# 备用方法：使用scikit-learn的Lasso（如果可用）
def solve_lasso_sklearn(X: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """
    使用scikit-learn的Lasso求解
    """
    try:
        from sklearn.linear_model import Lasso
        
        # 调整alpha参数
        alpha = gamma / (2 * X.shape[0])
        
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
        lasso.fit(X, y.flatten())
        
        return lasso.coef_.reshape(-1, 1)
        
    except ImportError:
        # 如果没有sklearn，使用自定义实现
        return solve_lasso(X, y, gamma)




