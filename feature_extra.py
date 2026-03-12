import numpy as np
from sklearn.decomposition import PCA
from typing import List, Tuple
from typing import Optional
import warnings
from typing import Dict
from collections import defaultdict
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
import itertools

def mmi_feature_extraction(X: np.ndarray, labels: np.ndarray, n_components: int, 
                          max_iter: int = 100, learning_rate: float = 0.01, 
                          tol: float = 1e-6) -> np.ndarray:
    """
    MMI（互信息最大化）线性特征提取（AR算法）
    :param X: 输入样本矩阵 (n_samples, n_features)
    :param labels: 样本标签
    :param n_components: 输出特征维数
    :param max_iter: 最大迭代次数
    :param learning_rate: 学习率
    :param tol: 收敛容限
    :return: 特征提取矩阵 W (n_components, n_features)
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(labels))
    
    # 1. 对x做球化PCA变换
    pca = PCA(n_components=None, whiten=True)  # whiten=True实现球化
    X_white = pca.fit_transform(X)
    
    # 计算每个类别的先验概率
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    p_c = label_counts / n_samples
    
    # 预计算类别条件样本
    class_samples = []
    for label in unique_labels:
        class_mask = (labels == label)
        class_samples.append(X_white[class_mask])
    
    # 定义负熵估计的常数
    k1 = 36 / (8 * np.sqrt(3) - 9)
    k2 = 24 / (16 * np.sqrt(3) - 27)
    sqrt_half = np.sqrt(0.5)
    
    def compute_neg_entropy(y: np.ndarray) -> float:
        """计算负熵 J(y) - 公式(8.9)"""
        exp_term = np.exp(-y**2 / 2)
        E_exp = np.mean(exp_term)
        J = k1 * (np.mean(y * exp_term))**2 + k2 * (E_exp - sqrt_half)**2
        return J
    
    def compute_neg_entropy_gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """计算负熵的梯度 ∇_w J(y) - 公式(8.11)"""
        exp_term = np.exp(-y**2 / 2)
        y_exp_term = y * exp_term
        y_squared_exp_term = (1 - y**2) * exp_term
        
        E_y_exp = np.mean(y_exp_term)
        E_y_squared_exp_x = np.mean(y_squared_exp_term.reshape(-1, 1) * x, axis=0)
        E_y_exp_x = np.mean(y_exp_term.reshape(-1, 1) * x, axis=0)
        E_exp = np.mean(exp_term)
        
        gradient = (2 * k1 * E_y_exp * E_y_squared_exp_x - 
                   2 * k2 * (E_exp - sqrt_half) * E_y_exp_x)
        return gradient
    
    def compute_mmi_gradient(w: np.ndarray, X: np.ndarray, 
                            class_samples: list, p_c: np.ndarray) -> np.ndarray:
        """计算MMI的梯度 - 公式(8.10)"""
        # 整体投影
        y_all = X @ w
        
        # 整体负熵梯度
        grad_J_all = compute_neg_entropy_gradient(y_all, X)
        
        # 初始化条件负熵梯度和方差项
        grad_J_cond_sum = np.zeros_like(w)
        variance_term_sum = np.zeros_like(w)
        
        for k, (c_samples, prob) in enumerate(zip(class_samples, p_c)):
            # 类别条件投影
            y_c = c_samples @ w
            
            # 类别条件负熵梯度
            grad_J_cond = compute_neg_entropy_gradient(y_c, c_samples)
            grad_J_cond_sum += prob * grad_J_cond
            
            # 方差项：C_{y|c_k} w / (w^T C_{y|c_k} w)
            var_y_c = np.var(y_c)
            if var_y_c > 1e-10:
                y_c_centered = y_c - np.mean(y_c)
                variance_gradient = 2 * np.mean(y_c_centered.reshape(-1, 1) * c_samples, axis=0) / var_y_c
                variance_term_sum += prob * variance_gradient
        
        # 公式(8.10)
        gradient = grad_J_cond_sum - grad_J_all - variance_term_sum
        return gradient
    
    def compute_mutual_information(w: np.ndarray) -> float:
        """计算I(c, y)用于监控收敛"""
        y_all = X_white @ w
        H_y = 0.5 * np.log(2 * np.pi * np.e * np.var(y_all))
        H_y_cond = 0
        for k, c_samples in enumerate(class_samples):
            y_c = c_samples @ w
            H_y_cond += p_c[k] * 0.5 * np.log(2 * np.pi * np.e * np.var(y_c))
        return H_y - H_y_cond
    
    # 初始化投影向量列表
    w_vectors = []
    
    # 2. 第一个分量：无正交约束，只有单位范数约束
    print("提取第1个特征分量...")
    w1 = np.random.randn(n_features)
    w1 = w1 / np.linalg.norm(w1)
    
    prev_I = -np.inf
    for iteration in range(max_iter):
        # 计算梯度
        grad = compute_mmi_gradient(w1, X_white, class_samples, p_c)
        
        # 更新
        w1_new = w1 + learning_rate * grad
        
        # 单位范数约束
        w1_new = w1_new / np.linalg.norm(w1_new)
        
        # 检查收敛
        current_I = compute_mutual_information(w1_new)
        if abs(current_I - prev_I) < tol:
            w1 = w1_new
            break
        
        w1 = w1_new
        prev_I = current_I
    
    w_vectors.append(w1)
    
    # 3. 后续分量：单位范数 + 正交约束
    for i in range(1, n_components):
        print(f"提取第{i+1}个特征分量...")
        
        # 随机初始化，并与已有分量正交化
        wi = np.random.randn(n_features)
        
        # Gram-Schmidt正交化
        for wj in w_vectors:
            wi = wi - np.dot(wi, wj) * wj
        
        # 单位范数
        wi = wi / np.linalg.norm(wi)
        
        prev_I = -np.inf
        for iteration in range(max_iter):
            # 计算梯度
            grad = compute_mmi_gradient(wi, X_white, class_samples, p_c)
            
            # 更新
            wi_new = wi + learning_rate * grad
            
            # 正交化处理（Gram-Schmidt）
            for wj in w_vectors:
                wi_new = wi_new - np.dot(wi_new, wj) * wj
            
            # 单位范数约束
            norm = np.linalg.norm(wi_new)
            if norm > 1e-10:
                wi_new = wi_new / norm
            else:
                # 如果norm太小，重新随机初始化
                wi_new = np.random.randn(n_features)
                for wj in w_vectors:
                    wi_new = wi_new - np.dot(wi_new, wj) * wj
                wi_new = wi_new / np.linalg.norm(wi_new)
            
            # 检查收敛
            current_I = compute_mutual_information(wi_new)
            if abs(current_I - prev_I) < tol:
                wi = wi_new
                break
            
            wi = wi_new
            prev_I = current_I
        
        w_vectors.append(wi)
    
    # 4. 构建变换矩阵 W
    W = np.array(w_vectors)
    
    return W

def lda_feature_extraction(X: np.ndarray, labels: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
    """
    LDA线性鉴别分析特征提取
    :param X: 输入样本矩阵 (n_samples, n_features)
    :param labels: 样本标签 (n_samples,)
    :param n_components: 输出特征维数（≤ C-1），默认为C-1
    :return: 特征提取矩阵 W (n_components, n_features)
    """
    # 获取基本统计信息
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # 检查输入有效性
    if n_classes < 2:
        warnings.warn("类别数小于2，LDA无法执行，返回单位矩阵")
        return np.eye(min(n_features, n_components)) if n_components else np.eye(n_features)
    
    # 设置输出维数（不能超过类别数-1）
    max_components = min(n_features, n_classes - 1)
    if n_components is None:
        n_components = max_components
    else:
        if n_components > max_components:
            warnings.warn(f"n_components不能超过类别数-1({max_components})，已自动调整为{max_components}")
            n_components = max_components
    
    # 计算总体均值向量 m (公式8.18)
    overall_mean = np.mean(X, axis=0)
    
    # 初始化类内散布矩阵和类间散布矩阵
    S_w = np.zeros((n_features, n_features))  # 类内散布矩阵 (公式8.14)
    S_b = np.zeros((n_features, n_features))  # 类间散布矩阵 (公式8.17)
    
    # 逐类计算
    for label in unique_labels:
        # 获取当前类别的样本
        class_mask = (labels == label)
        class_samples = X[class_mask]
        n_i = len(class_samples)  # N_i: 第i类的样本数目
        
        # 计算类别均值向量 m_i (公式8.16)
        # 注意：公式中应该是 1/N_i * sum(x)，修正了原文可能的笔误
        class_mean = np.mean(class_samples, axis=0)
        
        # 计算类内散布矩阵 S_i (公式8.15)
        # 对当前类别的每个样本，计算 (x - m_i)(x - m_i)^T
        centered_class = class_samples - class_mean
        S_i = centered_class.T @ centered_class  # (n_features, n_features)
        
        # 累加到类内散布矩阵 (公式8.14)
        S_w += S_i
        
        # 计算类间散布矩阵贡献 (公式8.17)
        # n_i * (m_i - m)(m_i - m)^T
        mean_diff = (class_mean - overall_mean).reshape(-1, 1)
        S_b += n_i * (mean_diff @ mean_diff.T)
    
    # 添加正则化项以避免奇异矩阵
    reg_term = 1e-6 * np.eye(n_features)
    S_w_reg = S_w + reg_term
    
    try:
        # 求解广义特征值问题：S_b * w = lambda * S_w * w (公式8.20)
        # 方法1：使用scipy.linalg.eigh求解广义特征值问题
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(S_b, S_w_reg)
        
        # 特征值已经是按升序排列，我们想要最大的几个特征值对应的特征向量
        # 所以取最后n_components个
        idx = np.argsort(eigenvalues)[::-1]  # 降序排列
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 取前n_components个特征向量
        W = eigenvectors[:, :n_components].T
        
    except ImportError:
        # 方法2：如果没有scipy，使用标准方法转换为普通特征值问题
        # 计算 S_w^(-1) * S_b
        try:
            S_w_inv = np.linalg.inv(S_w_reg)
            A = S_w_inv @ S_b
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            # 只取实部（理论上特征值应为实数）
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            
            # 按特征值降序排列
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # 取前n_components个特征向量
            W = eigenvectors[:, :n_components].T
            
        except np.linalg.LinAlgError:
            warnings.warn("矩阵奇异，使用PCA替代")
            # 如果S_w奇异，使用PCA作为后备方案
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            pca.fit(X)
            W = pca.components_
    
    # 确保输出维数正确
    if W.shape[0] != n_components:
        W = W[:n_components]
    
    return W

def sda_feature_extraction(X: np.ndarray, labels: np.ndarray, H_max: int = 10) -> Tuple[np.ndarray, dict]:
    """
    SDA子类鉴别分析特征提取
    :param X: 输入样本矩阵 (n_samples, n_features)
    :param labels: 样本标签 (n_samples,)
    :param H_max: 最大子类数搜索范围
    :return: 特征提取矩阵 W，子类划分信息
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        warnings.warn("类别数小于2，SDA无法执行，返回单位矩阵")
        return np.eye(n_features), {}
    
    # (1) 采用最近邻聚类方法对各类样本按距离排序
    print("步骤1: 对各类样本进行最近邻排序...")
    class_sorted_indices = {}
    class_samples_dict = {}
    
    for label in unique_labels:
        class_mask = (labels == label)
        class_samples = X[class_mask]
        class_samples_dict[label] = class_samples
        
        # 使用最近邻对样本进行排序
        if len(class_samples) > 1:
            nbrs = NearestNeighbors(n_neighbors=min(len(class_samples), 5), 
                                    metric='euclidean').fit(class_samples)
            # 从第一个样本开始，构建排序路径
            sorted_idx = [0]  # 从第一个样本开始
            remaining = set(range(1, len(class_samples)))
            
            while remaining:
                current = sorted_idx[-1]
                # 找到最近的未访问样本
                distances, indices = nbrs.kneighbors(class_samples[current].reshape(1, -1))
                for idx in indices[0]:
                    if idx in remaining:
                        sorted_idx.append(idx)
                        remaining.remove(idx)
                        break
                else:
                    # 如果没有找到，取任意剩余样本
                    sorted_idx.append(remaining.pop())
            
            class_sorted_indices[label] = sorted_idx
        else:
            class_sorted_indices[label] = [0]
    
    # (2) 计算类内散布矩阵 S_w
    print("步骤2: 计算类内散布矩阵 S_w...")
    S_w = np.zeros((n_features, n_features))
    
    for label in unique_labels:
        class_samples = class_samples_dict[label]
        class_mean = np.mean(class_samples, axis=0)
        centered = class_samples - class_mean
        S_w += centered.T @ centered
    
    # 添加正则化项
    reg_term = 1e-6 * np.eye(n_features)
    S_w_reg = S_w + reg_term
    
    # 计算各类样本数
    class_sizes = {label: len(class_samples_dict[label]) for label in unique_labels}
    min_class_size = min(class_sizes.values())
    total_samples = n_samples
    
    # 存储每个H对应的结果
    phi_values = {}
    best_H = None
    best_phi = float('inf')
    best_W = None
    best_subclass_info = None
    
    # (3) 对 H = 1, ..., H_max 执行搜索
    print(f"步骤3: 搜索最优子类数 H (1-{H_max})...")
    
    for H in range(1, H_max + 1):
        print(f"  尝试 H = {H}...")
        
        # ① 根据式(8.24)计算每类的子类数
        subclass_counts = {}
        for label in unique_labels:
            # H_i = fix(H * N_i / min{N_i})
            subclass_counts[label] = max(1, int(H * class_sizes[label] / min_class_size))
        
        # ② 对每类样本进行均匀子类划分
        subclasses = {}  # 存储每个子类的样本索引
        subclass_means = {}  # 存储每个子类的均值
        subclass_sizes = {}  # 存储每个子类的样本数
        
        for label in unique_labels:
            sorted_idx = class_sorted_indices[label]
            n_subclasses = subclass_counts[label]
            class_samples = class_samples_dict[label]
            
            # 均匀划分
            indices_split = np.array_split(sorted_idx, n_subclasses)
            subclasses[label] = []
            
            for sub_idx, sub_indices in enumerate(indices_split):
                sub_key = (label, sub_idx)
                sub_samples = class_samples[sub_indices]
                subclass_means[sub_key] = np.mean(sub_samples, axis=0)
                subclass_sizes[sub_key] = len(sub_indices)
                subclasses[label].append(sub_key)
        
        # ③ 根据式(8.22)计算 S_b
        S_b = np.zeros((n_features, n_features))
        
        # 获取所有子类键的列表
        all_subclasses = []
        for label in unique_labels:
            all_subclasses.extend(subclasses[label])
        
        # 计算所有子类对之间的贡献
        for i in range(len(all_subclasses)):
            sub_key_i = all_subclasses[i]
            label_i, sub_idx_i = sub_key_i
            
            for j in range(i + 1, len(all_subclasses)):
                sub_key_j = all_subclasses[j]
                label_j, sub_idx_j = sub_key_j
                
                # 只考虑不同类的子类对
                if label_i != label_j:
                    # 计算权重 (N_ij/N) * (N_kl/N)
                    weight = (subclass_sizes[sub_key_i] / total_samples) * \
                             (subclass_sizes[sub_key_j] / total_samples)
                    
                    # 计算均值差
                    mean_diff = subclass_means[sub_key_i] - subclass_means[sub_key_j]
                    mean_diff = mean_diff.reshape(-1, 1)
                    
                    # 累加贡献
                    S_b += weight * (mean_diff @ mean_diff.T)
        
        # ④ 对 S_b W = S_w W A 执行特征值分解
        try:
            # 求解广义特征值问题
            eigenvalues, eigenvectors = linalg.eigh(S_b, S_w_reg)
            
            # 只取实部
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
            
            # 按特征值降序排列
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # ⑤ 根据式(8.23)计算 Φ(H)
            # 确定 r < rank(S_b)
            rank_S_b = np.linalg.matrix_rank(S_b)
            r = min(rank_S_b - 1, len(eigenvalues) - 1) if rank_S_b > 1 else 1
            
            # 获取 S_w 的特征向量
            _, u = linalg.eigh(S_w_reg)
            u = np.real(u)
            
            # 计算 Φ(H)
            phi = 0
            count = 0
            for i in range(min(r, len(eigenvalues))):
                w_i = eigenvectors[:, i]
                for j in range(min(n_features, len(u))):
                    u_j = u[:, j]
                    projection = u_j.T @ w_i
                    phi += projection ** 2
                    count += 1
            
            if count > 0:
                phi = phi / (H * count)  # 除以 H * count 进行归一化
            else:
                phi = float('inf')
            
            phi_values[H] = phi
            
            # 更新最优值
            if phi < best_phi:
                best_phi = phi
                best_H = H
                best_W = eigenvectors[:, :min(n_features, n_classes - 1)].T
                best_subclass_info = {
                    'H': H,
                    'subclass_counts': subclass_counts,
                    'subclass_sizes': subclass_sizes
                }
                
        except Exception as e:
            print(f"    特征值分解失败: {e}")
            phi_values[H] = float('inf')
    
    # (4) 选择使 Φ(H) 最小的 H 值，重做子类划分
    print(f"\n步骤4: 选择最优 H = {best_H}")
    
    if best_H is None:
        warnings.warn("未找到有效的H值，使用PCA作为后备")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_features, n_classes - 1))
        pca.fit(X)
        return pca.components_, {}
    
    # 使用最优H重新计算子类划分
    subclass_counts = {}
    for label in unique_labels:
        subclass_counts[label] = max(1, int(best_H * class_sizes[label] / min_class_size))
    
    # (5) 重计算 S_b 并执行特征值分解，获得最终的特征提取映射 W
    print("步骤5: 使用最优H重新计算并获取最终特征提取矩阵...")
    
    # 重新划分子类
    subclasses = {}
    subclass_means = {}
    subclass_sizes = {}
    
    for label in unique_labels:
        sorted_idx = class_sorted_indices[label]
        n_subclasses = subclass_counts[label]
        class_samples = class_samples_dict[label]
        
        indices_split = np.array_split(sorted_idx, n_subclasses)
        subclasses[label] = []
        
        for sub_idx, sub_indices in enumerate(indices_split):
            sub_key = (label, sub_idx)
            sub_samples = class_samples[sub_indices]
            subclass_means[sub_key] = np.mean(sub_samples, axis=0)
            subclass_sizes[sub_key] = len(sub_indices)
            subclasses[label].append(sub_key)
    
    # 重新计算 S_b
    S_b_final = np.zeros((n_features, n_features))
    all_subclasses = []
    for label in unique_labels:
        all_subclasses.extend(subclasses[label])
    
    for i in range(len(all_subclasses)):
        sub_key_i = all_subclasses[i]
        label_i, _ = sub_key_i
        
        for j in range(i + 1, len(all_subclasses)):
            sub_key_j = all_subclasses[j]
            label_j, _ = sub_key_j
            
            if label_i != label_j:
                weight = (subclass_sizes[sub_key_i] / total_samples) * \
                         (subclass_sizes[sub_key_j] / total_samples)
                mean_diff = subclass_means[sub_key_i] - subclass_means[sub_key_j]
                mean_diff = mean_diff.reshape(-1, 1)
                S_b_final += weight * (mean_diff @ mean_diff.T)
    
    # 最终特征值分解
    eigenvalues, eigenvectors = linalg.eigh(S_b_final, S_w_reg)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 构建最终的特征提取矩阵
    n_components = min(n_features, n_classes - 1)
    W_final = eigenvectors[:, :n_components].T
    
    # 构建返回信息
    result_info = {
        'best_H': best_H,
        'phi_values': phi_values,
        'subclass_counts': subclass_counts,
        'subclass_sizes': {f"{k[0]}_{k[1]}": v for k, v in subclass_sizes.items()},
        'n_classes': n_classes,
        'n_features': n_features
    }
    
    return W_final, result_info

import numpy as np
from typing import Tuple, List, Dict
from scipy import linalg
from scipy.stats import kurtosis
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import warnings

def sda_improved(X: np.ndarray, labels: np.ndarray, H_max: int = 10) -> np.ndarray:
    """
    SDA改进方法：子类划分 + 修正类内散布矩阵
    :param X: 输入样本矩阵 (n_samples, n_features)
    :param labels: 样本标签 (n_samples,)
    :param H_max: 最大子类数
    :return: 特征提取矩阵 W (n_components, n_features)
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        warnings.warn("类别数小于2，SDA无法执行，返回单位矩阵")
        return np.eye(min(n_features, n_classes - 1))
    
    print("步骤1: 基于负熵选择高分裂特征维度...")
    
    # (1) 依据负熵选择高分裂特征维度
    # 计算每个特征的负熵（使用近似公式）
    def compute_neg_entropy(feature: np.ndarray) -> float:
        """计算特征的负熵近似值"""
        # 标准化特征
        feature_std = (feature - np.mean(feature)) / np.std(feature)
        
        # 使用kurtosis作为非高斯性的度量（负熵的近似）
        # 高斯分布的峰度为3，所以用|峰度-3|作为分裂程度的度量
        kurt = kurtosis(feature_std)
        neg_entropy = np.abs(kurt - 3)
        
        return neg_entropy
    
    # 计算每个特征的负熵
    feature_neg_entropy = []
    for i in range(n_features):
        neg_entropy = compute_neg_entropy(X[:, i])
        feature_neg_entropy.append(neg_entropy)
    
    # 选择负熵较大的特征（前50%或至少10个）
    feature_neg_entropy = np.array(feature_neg_entropy)
    threshold = np.percentile(feature_neg_entropy, 50)  # 中位数阈值
    high_split_features = np.where(feature_neg_entropy >= threshold)[0]
    
    # 如果选出的特征太少，取前10个
    if len(high_split_features) < min(10, n_features):
        high_split_features = np.argsort(feature_neg_entropy)[-min(10, n_features):]
    
    print(f"  选择了 {len(high_split_features)} 个高分裂特征维度")
    
    # (2) 使用层次聚类方法将筛选出的低维特征集聚类为M个子类
    print("步骤2: 层次聚类划分子类...")
    
    # 提取高分裂特征
    X_selected = X[:, high_split_features]
    
    # 标准化
    scaler = StandardScaler()
    X_selected_scaled = scaler.fit_transform(X_selected)
    
    # 对每个类别单独进行聚类
    subclass_info = {}  # 存储每个样本的子类标签
    subclass_means = {}  # 存储每个子类的均值
    subclass_sizes = {}  # 存储每个子类的样本数
    subclass_samples = {}  # 存储每个子类的样本索引
    
    for label in unique_labels:
        class_mask = (labels == label)
        class_samples_idx = np.where(class_mask)[0]
        class_data = X_selected_scaled[class_mask]
        
        n_class_samples = len(class_samples_idx)
        
        # 对该类进行层次聚类
        # 最大聚类数不超过样本数的一半和H_max
        max_clusters = min(H_max, max(2, n_class_samples // 2))
        
        if n_class_samples >= max_clusters:
            # 使用层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=max_clusters,
                metric='euclidean',
                linkage='ward'
            )
            cluster_labels = clustering.fit_predict(class_data)
            
            # 记录子类信息
            for cluster_id in range(max_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                cluster_indices = class_samples_idx[cluster_mask]
                sub_key = (label, cluster_id)
                
                # 计算子类均值（在原始特征空间）
                sub_mean = np.mean(X[cluster_indices], axis=0)
                subclass_means[sub_key] = sub_mean
                subclass_sizes[sub_key] = len(cluster_indices)
                subclass_samples[sub_key] = cluster_indices
                
                for idx in cluster_indices:
                    subclass_info[idx] = sub_key
        else:
            # 样本太少，整个类作为一个子类
            sub_key = (label, 0)
            sub_mean = np.mean(X[class_mask], axis=0)
            subclass_means[sub_key] = sub_mean
            subclass_sizes[sub_key] = n_class_samples
            subclass_samples[sub_key] = class_samples_idx
            
            for idx in class_samples_idx:
                subclass_info[idx] = sub_key
    
    print(f"  总共划分出 {len(subclass_means)} 个子类")
    
    # (3) 根据式(8.25)计算类内散布矩阵 S_w
    print("步骤3: 计算修正的类内散布矩阵 S_w...")
    
    S_w = np.zeros((n_features, n_features))
    
    for sub_key, indices in subclass_samples.items():
        label, sub_idx = sub_key
        sub_mean = subclass_means[sub_key]
        
        # 对该子类内的所有样本
        for idx in indices:
            x = X[idx]
            diff = (x - sub_mean).reshape(-1, 1)
            S_w += diff @ diff.T
    
    # 添加正则化项
    reg_term = 1e-6 * np.eye(n_features)
    S_w_reg = S_w + reg_term
    
    # (4) 根据式(8.22)计算 S_b
    print("步骤4: 计算修正的类间散布矩阵 S_b...")
    
    S_b = np.zeros((n_features, n_features))
    total_samples = n_samples
    
    # 获取所有子类键的列表
    all_subclasses = list(subclass_means.keys())
    
    # 计算所有跨类子类对的贡献
    for i in range(len(all_subclasses)):
        sub_key_i = all_subclasses[i]
        label_i, _ = sub_key_i
        
        for j in range(i + 1, len(all_subclasses)):
            sub_key_j = all_subclasses[j]
            label_j, _ = sub_key_j
            
            # 只考虑不同类的子类对
            if label_i != label_j:
                # 计算权重 (N_ij/N) * (N_kl/N)
                weight = (subclass_sizes[sub_key_i] / total_samples) * \
                         (subclass_sizes[sub_key_j] / total_samples)
                
                # 计算均值差
                mean_diff = subclass_means[sub_key_i] - subclass_means[sub_key_j]
                mean_diff = mean_diff.reshape(-1, 1)
                
                # 累加贡献
                S_b += weight * (mean_diff @ mean_diff.T)
    
    # 对 S_b W = S_w W A 执行特征值分解计算
    print("步骤5: 执行特征值分解，获取特征提取映射...")
    
    try:
        # 求解广义特征值问题
        eigenvalues, eigenvectors = linalg.eigh(S_b, S_w_reg)
        
        # 只取实部
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # 构建特征提取矩阵（最多取C-1维）
        n_components = min(n_features, n_classes - 1)
        W = eigenvectors[:, :n_components].T
        
    except Exception as e:
        print(f"  特征值分解失败: {e}")
        # 失败时使用PCA作为后备
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_features, n_classes - 1))
        pca.fit(X)
        W = pca.components_
    
    return W