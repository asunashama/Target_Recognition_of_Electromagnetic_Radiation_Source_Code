import numpy as np
from typing import List,Tuple
from scipy.signal import czt

def align_time(samples: List[np.ndarray], snr_order: bool = True) -> List[np.ndarray]:
    """
    基于互相关的时间对齐
    :param samples: 信号样本列表，每个为N×1向量
    :param snr_order: 是否按信噪比从大到小重排
    :return: 时间对齐后的样本列表
    """
    # 复制样本列表以避免修改原始数据
    aligned_samples = [sample.copy() for sample in samples]
    M = len(aligned_samples)
    
    if M <= 1:
        return aligned_samples
    
    # (1) 按信噪比从大到小重排信号样本
    if snr_order:
        # 计算每个样本的信噪比（使用信号功率与噪声功率的比值）
        snrs = []
        for sample in aligned_samples:
            # 简单估计：使用信号的方差作为信号功率，假设噪声均值为0
            signal_power = np.var(sample)
            # 添加一个小值避免除零
            snr = signal_power / (np.var(sample - np.mean(sample)) + 1e-10)
            snrs.append(snr)
        
        # 按信噪比降序排序
        sorted_indices = np.argsort(snrs)[::-1]
        aligned_samples = [aligned_samples[i] for i in sorted_indices]
    
    N = len(aligned_samples[0])  # 信号长度
    
    # (2) 对k=1,...,M执行对齐
    for k in range(1, M):
        # ① 计算与前面所有已对齐信号的互相关
        # 初始化累计互相关数组
        sum_correlation = np.zeros(N)
        
        for j in range(k):
            # 对每个时延τ计算互相关
            for tau in range(N):
                # 对样本z_k进行τ点循环移位
                shifted_k = np.roll(aligned_samples[k], tau)
                # 计算互相关 |z_j^H(0)z_k(τ)|
                # 注意：z_j^H(0)表示z_j的共轭转置
                correlation = np.abs(np.dot(aligned_samples[j].conj().T, shifted_k))
                sum_correlation[tau] += correlation
        
        # ② 找到使累计互相关最大的时延
        best_tau = np.argmax(sum_correlation)
        
        # ③ 对第k个脉冲进行best_tau点循环移位
        aligned_samples[k] = np.roll(aligned_samples[k], best_tau)
    
    return aligned_samples

def align_frequency(samples: List[np.ndarray], snr_order: bool = True) -> List[np.ndarray]:
    """
    基于互相关的频率对齐
    :param samples: 时间对齐后的样本
    :param snr_order: 是否按信噪比重排
    :return: 频率对齐后的样本
    """
    # 复制样本列表以避免修改原始数据
    aligned_samples = [sample.copy() for sample in samples]
    M = len(aligned_samples)
    
    if M <= 1:
        return aligned_samples
    
    # (1) 按信噪比从大到小重排信号样本
    if snr_order:
        # 计算每个样本的信噪比
        snrs = []
        for sample in aligned_samples:
            # 简单估计：使用信号的方差作为信号功率
            signal_power = np.var(sample)
            # 估计噪声功率（假设噪声均值为0）
            noise_power = np.var(sample - np.mean(sample)) + 1e-10
            snr = signal_power / noise_power
            snrs.append(snr)
        
        # 按信噪比降序排序
        sorted_indices = np.argsort(snrs)[::-1]
        aligned_samples = [aligned_samples[i] for i in sorted_indices]
    
    N = len(aligned_samples[0])  # 信号长度
    
    # 设置CZT参数
    # 在单位圆上均匀采样，采样点数可以设置为信号长度的2倍以获得更好的频率分辨率
    m = N * 2  # CZT输出点数
    # 频率搜索范围：[-π, π]
    w = np.linspace(-np.pi, np.pi, m)
    
    # (2) 对k=1,...,M执行频率对齐
    for k in range(1, M):
        # 初始化累计功率谱
        sum_power_spectrum = np.zeros(m, dtype=float)
        
        # ① 对所有j=1,...,k-1，计算z_j*(n)z_k(n)的CZT谱
        for j in range(k):
            # 计算z_j*(n)z_k(n)
            # 对于复数信号，需要取共轭
            product = np.conj(aligned_samples[j]) * aligned_samples[k]
            
            # 计算CZT谱
            # 使用scipy的czt函数，在单位圆上计算Z变换
            czt_spectrum = czt(product, m=m)
            
            # 存储C_j(ω)的幅值平方
            # 注意：算法中需要的是|C_j(ω)|^2
            power_spectrum = np.abs(czt_spectrum) ** 2
            sum_power_spectrum += power_spectrum
        
        # ② 找到使累计功率谱最大的频率
        best_idx = np.argmax(sum_power_spectrum)
        best_omega = w[best_idx]
        
        # ③ 对第k个脉冲按照best_omega去相差
        # 构建相位校正因子：e^{-j*ω*n}
        n = np.arange(N).reshape(-1, 1)  # 保持为列向量
        correction_factor = np.exp(-1j * best_omega * n)
        
        # 应用相位校正
        aligned_samples[k] = aligned_samples[k] * correction_factor
    
    return aligned_samples

def align_phase(samples: List[np.ndarray], snr_order: bool = True) -> List[np.ndarray]:
    """
    基于互相关的相位对齐
    :param samples: 频率对齐后的样本
    :param snr_order: 是否按信噪比重排
    :return: 相位对齐后的样本
    """
    # 复制样本列表以避免修改原始数据
    aligned_samples = [sample.copy() for sample in samples]
    M = len(aligned_samples)
    
    if M <= 1:
        return aligned_samples
    
    # (1) 按信噪比从大到小重排信号样本
    if snr_order:
        # 计算每个样本的信噪比
        snrs = []
        for sample in aligned_samples:
            # 简单估计：使用信号的方差作为信号功率
            signal_power = np.var(sample)
            # 估计噪声功率（假设噪声均值为0）
            noise_power = np.var(sample - np.mean(sample)) + 1e-10
            snr = signal_power / noise_power
            snrs.append(snr)
        
        # 按信噪比降序排序
        sorted_indices = np.argsort(snrs)[::-1]
        aligned_samples = [aligned_samples[i] for i in sorted_indices]
    
    N = len(aligned_samples[0])  # 信号长度
    
    # (2) 对k=1,...,M执行相位对齐
    for k in range(1, M):
        # 初始化累计互相关和
        total_correlation = 0j
        
        # ① 对所有j=1,...,k-1，计算c_j(n) = z_j*(n)z_k(n)
        for j in range(k):
            # 计算z_j*(n)z_k(n)并求和
            # 注意：公式中是从n=1到N-1，排除n=0（如果N足够大，包含n=0影响不大）
            # 严格遵循公式，使用索引1到N-1
            for n in range(1, N):
                # 计算c_j(n)
                c_j_n = np.conj(aligned_samples[j][n]) * aligned_samples[k][n]
                total_correlation += c_j_n
        
        # ② 求总和的相位
        # 使用np.angle获取复数的相位角（弧度）
        best_phi = np.angle(total_correlation)
        
        # ③ 对第k个脉冲按照best_phi去相差
        # 构建相位校正因子：e^{-j*φ}
        correction_factor = np.exp(-1j * best_phi)
        
        # 应用相位校正（对所有采样点应用相同的相位校正）
        aligned_samples[k] = aligned_samples[k] * correction_factor
    
    return aligned_samples

def extract_common_waveform(samples: List[np.ndarray], beta: float = 0.001, kappa: float = 1.345) -> np.ndarray:
    """
    提取同源样本的公共波形
    :param samples: 同源信号样本列表
    :param beta: 收敛阈值
    :param kappa: 权值门限
    :return: 公共波形向量
    """
    # 复制样本列表以避免修改原始数据
    sample_list = [sample.copy().flatten() for sample in samples]
    Q = len(sample_list)  # 样本数量
    N = len(sample_list[0])  # 信号长度
    
    if Q == 0:
        return np.array([])
    if Q == 1:
        return sample_list[0]
    
    # 1. 初始化权值 ω_k = 1
    weights = np.ones(Q)
    
    # 初始化公共波形（使用第一个样本作为初始估计）
    mu = sample_list[0].copy()
    
    # 初始化幅度估计
    A_est = np.ones(Q)
    
    # 初始化频率偏移估计（假设已经过频率对齐，偏移很小）
    # 这里简单假设频率偏移为0，实际应用中可能需要更精细的估计
    v_est = np.zeros(Q)
    
    # 计算初始损失函数值
    def compute_loss(mu, A, v, weights):
        loss = 0
        for k in range(Q):
            # 构建频率偏移矩阵 Ω(v_k)
            # Ω(v_k) 是对角矩阵，对角元为 exp(-j*n*v_k)
            n = np.arange(N)
            Omega = np.diag(np.exp(-1j * n * v[k]))
            
            # 计算重建误差
            reconstruction = A[k] * Omega @ mu
            error = sample_list[k] - reconstruction
            loss += weights[k] * np.linalg.norm(error) ** 2
        return loss
    
    L_prev = compute_loss(mu, A_est, v_est, weights)
    
    i = 1
    max_iter = 100  # 最大迭代次数，防止无限循环
    
    while i < max_iter:
        i += 1
        
        # 2. 求解最小化问题，更新公共波形和幅度
        # 对于固定的 v，可以通过最小二乘求解 mu 和 A
        
        # 先固定 v，求解 mu 和 A
        # 构建方程组：对于每个 k，有 z_k = A_k * Ω(v_k) * μ
        # 可以写成矩阵形式
        
        # 方法：使用交替最小化
        # 先固定 A 更新 mu，再固定 mu 更新 A
        
        # 更新 mu（公共波形）
        # 最小二乘解：mu = (∑ ω_k A_k^2 Ω^H Ω)^{-1} (∑ ω_k A_k Ω^H z_k)
        # 由于 Ω^H Ω = I，简化计算
        
        numerator_mu = np.zeros(N, dtype=complex)
        denominator_mu = 0
        
        for k in range(Q):
            n = np.arange(N)
            Omega = np.diag(np.exp(-1j * n * v_est[k]))
            
            # 加权贡献
            numerator_mu += weights[k] * A_est[k] * Omega.conj().T @ sample_list[k]
            denominator_mu += weights[k] * A_est[k] ** 2
        
        if denominator_mu > 0:
            mu = numerator_mu / denominator_mu
        
        # 更新 A（幅度）
        for k in range(Q):
            n = np.arange(N)
            Omega = np.diag(np.exp(-1j * n * v_est[k]))
            
            # 最小二乘解：A_k = (μ^H Ω^H z_k) / (μ^H μ)
            # 假设 μ^H μ 归一化
            mu_norm = np.linalg.norm(mu) ** 2
            if mu_norm > 0:
                A_est[k] = np.real((mu.conj().T @ (Omega.conj().T @ sample_list[k])) / mu_norm)
            else:
                A_est[k] = 0
        
        # 3. 计算权值
        sigma = 1.0  # 噪声标准差估计，实际应用中可能需要更精确的估计
        for k in range(Q):
            n = np.arange(N)
            Omega = np.diag(np.exp(-1j * n * v_est[k]))
            
            # 计算重建误差
            reconstruction = A_est[k] * Omega @ mu
            error = sample_list[k] - reconstruction
            error_norm = np.linalg.norm(error)
            
            # 计算 x = ||error|| / (σ√N)
            x = error_norm / (sigma * np.sqrt(N))
            
            # 应用权值函数 h(x)
            if np.abs(x) < kappa:
                weights[k] = 1.0
            else:
                weights[k] = kappa / np.abs(x)
        
        # 4. 计算新的损失函数值
        L_curr = compute_loss(mu, A_est, v_est, weights)
        
        # 检查收敛条件
        if L_prev > 0 and L_curr > 0:
            if L_prev / L_curr < 1 + beta:
                break
        
        L_prev = L_curr
    
    # 返回公共波形（取实部，假设原始信号是实信号）
    # 如果信号是复数，可以返回复数形式
    return np.real(mu)