#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估指标计算模块

此模块提供了一系列用于评估交通预测模型性能的指标，包括：
- MAE (Mean Absolute Error): 平均绝对误差
- RMSE (Root Mean Squared Error): 均方根误差
- MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差
- R² (Coefficient of Determination): 决定系数
- MSE (Mean Squared Error): 均方误差
- MRE (Mean Relative Error): 平均相对误差
- WAPE (Weighted Absolute Percentage Error): 加权绝对百分比误差
- SMAPE (Symmetric Mean Absolute Percentage Error): 对称平均绝对百分比误差
"""

import numpy as np
from typing import Dict, List, Union, Tuple


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (MAE)
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        
    Returns:
        float: MAE值
    """
    # 处理NaN值
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return 0.0
    
    # 计算绝对误差
    absolute_error = np.abs(y_true - y_pred)
    
    # 返回平均值（只考虑非NaN值）
    return np.mean(absolute_error[mask])


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (RMSE)
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        
    Returns:
        float: RMSE值
    """
    # 处理NaN值
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return 0.0
    
    # 计算平方误差
    squared_error = np.square(y_true - y_pred)
    
    # 返回均方根
    return np.sqrt(np.mean(squared_error[mask]))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方误差 (MSE)
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        
    Returns:
        float: MSE值
    """
    # 处理NaN值
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return 0.0
    
    # 计算平方误差
    squared_error = np.square(y_true - y_pred)
    
    # 返回平均值
    return np.mean(squared_error[mask])


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                   epsilon: float = 1e-8) -> float:
    """
    计算平均绝对百分比误差 (MAPE)
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        epsilon: 避免除以零的小值
        
    Returns:
        float: MAPE值（百分比）
    """
    # 处理NaN值
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return 0.0
    
    # 避免除以零
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    # 添加小值以避免除以零
    denominator = np.where(np.abs(y_true_masked) < epsilon, epsilon, y_true_masked)
    
    # 计算绝对百分比误差
    percentage_error = np.abs((y_true_masked - y_pred_masked) / denominator) * 100
    
    # 返回平均值
    return np.mean(percentage_error)


def mean_relative_error(y_true: np.ndarray, y_pred: np.ndarray, 
                        epsilon: float = 1e-8) -> float:
    """
    计算平均相对误差 (MRE)
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        epsilon: 避免除以零的小值
        
    Returns:
        float: MRE值
    """
    # 处理NaN值
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return 0.0
    
    # 避免除以零
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    # 添加小值以避免除以零
    denominator = np.where(np.abs(y_true_masked) < epsilon, epsilon, y_true_masked)
    
    # 计算相对误差
    relative_error = np.abs((y_true_masked - y_pred_masked) / denominator)
    
    # 返回平均值
    return np.mean(relative_error)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                             epsilon: float = 1e-8) -> float:
    """
    计算对称平均绝对百分比误差 (SMAPE)
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        epsilon: 避免除以零的小值
        
    Returns:
        float: SMAPE值（百分比）
    """
    # 处理NaN值
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return 0.0
    
    # 避免除以零
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    # 添加小值以避免除以零
    denominator = np.abs(y_true_masked) + np.abs(y_pred_masked) + epsilon
    
    # 计算对称绝对百分比误差
    symmetric_error = 2.0 * np.abs(y_true_masked - y_pred_masked) / denominator * 100
    
    # 返回平均值
    return np.mean(symmetric_error)


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                       epsilon: float = 1e-8) -> float:
    """
    计算加权绝对百分比误差 (WAPE)
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        epsilon: 避免除以零的小值
        
    Returns:
        float: WAPE值（百分比）
    """
    # 处理NaN值
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return 0.0
    
    # 计算绝对误差总和和真实值总和
    absolute_error_sum = np.sum(np.abs(y_true[mask] - y_pred[mask]))
    y_true_sum = np.sum(np.abs(y_true[mask]))
    
    # 避免除以零
    if y_true_sum < epsilon:
        return 0.0
    
    # 返回加权绝对百分比误差
    return (absolute_error_sum / y_true_sum) * 100


def coefficient_of_determination(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算决定系数 (R²)
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        
    Returns:
        float: R²值
    """
    # 处理NaN值
    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return 0.0
    
    # 提取非NaN值
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    # 计算真实值的均值
    y_true_mean = np.mean(y_true_masked)
    
    # 计算总平方和 (TSS)
    total_sum_of_squares = np.sum(np.square(y_true_masked - y_true_mean))
    
    # 如果总平方和为零，说明所有值相同
    if total_sum_of_squares < 1e-10:
        return 1.0 if np.all(np.abs(y_pred_masked - y_true_mean) < 1e-10) else 0.0
    
    # 计算残差平方和 (RSS)
    residual_sum_of_squares = np.sum(np.square(y_true_masked - y_pred_masked))
    
    # 计算R²
    r2 = 1.0 - (residual_sum_of_squares / total_sum_of_squares)
    
    # 确保R²在有效范围内
    return max(min(r2, 1.0), 0.0)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        
    Returns:
        Dict[str, float]: 包含所有指标的字典
    """
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "mre": mean_relative_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "wape": weighted_absolute_percentage_error(y_true, y_pred),
        "r2": coefficient_of_determination(y_true, y_pred)
    }
    
    return metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    metrics_list: List[str] = None) -> Dict[str, float]:
    """
    根据指定的指标列表计算评估指标
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        metrics_list: 要计算的指标列表，如果为None则计算所有指标
        
    Returns:
        Dict[str, float]: 包含指定指标的字典
    """
    # 定义可用的指标函数
    available_metrics = {
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error,
        "mse": mean_squared_error,
        "mape": mean_absolute_percentage_error,
        "mre": mean_relative_error,
        "smape": symmetric_mean_absolute_percentage_error,
        "wape": weighted_absolute_percentage_error,
        "r2": coefficient_of_determination
    }
    
    # 如果未指定指标列表，计算所有指标
    if metrics_list is None:
        metrics_list = list(available_metrics.keys())
    
    # 验证指标名称
    for metric in metrics_list:
        if metric not in available_metrics:
            raise ValueError(f"不支持的指标: {metric}. 可用的指标有: {list(available_metrics.keys())}")
    
    # 计算指定的指标
    results = {}
    for metric in metrics_list:
        results[metric] = available_metrics[metric](y_true, y_pred)
    
    return results


def compute_time_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 metrics_list: List[str] = None) -> Dict[str, np.ndarray]:
    """
    计算每个时间步的评估指标
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        metrics_list: 要计算的指标列表
        
    Returns:
        Dict[str, np.ndarray]: 每个指标在各个时间步的值
    """
    # 确定预测长度维度
    if y_true.ndim == 3:
        # 形状: [batch_size, num_nodes, pred_len]
        pred_len = y_true.shape[2]
    elif y_true.ndim == 2:
        # 形状: [batch_size, pred_len]
        pred_len = y_true.shape[1]
    else:
        raise ValueError(f"不支持的输入维度: {y_true.ndim}")
    
    # 初始化结果字典
    time_horizon_metrics = {}
    
    # 计算每个时间步的指标
    for t in range(pred_len):
        if y_true.ndim == 3:
            y_true_t = y_true[:, :, t]
            y_pred_t = y_pred[:, :, t]
        else:
            y_true_t = y_true[:, t]
            y_pred_t = y_pred[:, t]
        
        # 计算当前时间步的指标
        step_metrics = compute_metrics(y_true_t, y_pred_t, metrics_list)
        
        # 更新结果字典
        for metric_name, metric_value in step_metrics.items():
            if metric_name not in time_horizon_metrics:
                time_horizon_metrics[metric_name] = []
            time_horizon_metrics[metric_name].append(metric_value)
    
    # 转换为numpy数组
    for metric_name in time_horizon_metrics:
        time_horizon_metrics[metric_name] = np.array(time_horizon_metrics[metric_name])
    
    return time_horizon_metrics


def compute_node_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               metrics_list: List[str] = None) -> Dict[str, np.ndarray]:
    """
    计算每个节点的评估指标
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        metrics_list: 要计算的指标列表
        
    Returns:
        Dict[str, np.ndarray]: 每个指标在各个节点的值
    """
    if y_true.ndim != 3:
        raise ValueError(f"输入必须是3维数组，当前维度: {y_true.ndim}")
    
    num_nodes = y_true.shape[1]
    
    # 初始化结果字典
    node_metrics = {}
    
    # 计算每个节点的指标
    for n in range(num_nodes):
        y_true_n = y_true[:, n, :]
        y_pred_n = y_pred[:, n, :]
        
        # 计算当前节点的指标
        node_step_metrics = compute_metrics(y_true_n, y_pred_n, metrics_list)
        
        # 更新结果字典
        for metric_name, metric_value in node_step_metrics.items():
            if metric_name not in node_metrics:
                node_metrics[metric_name] = []
            node_metrics[metric_name].append(metric_value)
    
    # 转换为numpy数组
    for metric_name in node_metrics:
        node_metrics[metric_name] = np.array(node_metrics[metric_name])
    
    return node_metrics


def compute_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray, 
                                 metric_name: str = "mae", 
                                 n_bootstrap: int = 1000, 
                                 confidence_level: float = 0.95) -> Dict[str, float]:
    """
    使用自助法计算指标的置信区间
    
    Args:
        y_true: 真实值，形状为 [batch_size, num_nodes, pred_len]
        y_pred: 预测值，形状与 y_true 相同
        metric_name: 要计算置信区间的指标名称
        n_bootstrap: 自助法重复次数
        confidence_level: 置信水平
        
    Returns:
        Dict[str, float]: 包含置信区间信息的字典
    """
    # 定义可用的指标函数
    available_metrics = {
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error,
        "mse": mean_squared_error,
        "mape": mean_absolute_percentage_error,
        "mre": mean_relative_error,
        "smape": symmetric_mean_absolute_percentage_error,
        "wape": weighted_absolute_percentage_error,
        "r2": coefficient_of_determination
    }
    
    if metric_name not in available_metrics:
        raise ValueError(f"不支持的指标: {metric_name}. 可用的指标有: {list(available_metrics.keys())}")
    
    # 获取指标函数
    metric_func = available_metrics[metric_name]
    
    # 获取样本大小
    n_samples = y_true.shape[0]
    
    # 存储自助法的结果
    bootstrap_results = []
    
    # 执行自助法
    for _ in range(n_bootstrap):
        # 生成随机索引（有放回）
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # 根据索引选择样本
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # 计算指标
        metric_value = metric_func(y_true_boot, y_pred_boot)
        bootstrap_results.append(metric_value)
    
    # 计算置信区间
    alpha = 1.0 - confidence_level
    lower_bound = np.percentile(bootstrap_results, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_results, 100 * (1 - alpha / 2))
    
    # 计算原始指标值
    original_metric = metric_func(y_true, y_pred)
    
    return {
        "metric": metric_name,
        "value": original_metric,
        "confidence_level": confidence_level,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "standard_error": np.std(bootstrap_results)
    }


def metrics_summary_table(y_true: np.ndarray, y_pred: np.ndarray, 
                          metrics_list: List[str] = None) -> str:
    """
    生成指标汇总表格
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        metrics_list: 要计算的指标列表
        
    Returns:
        str: 格式化的指标汇总表格
    """
    # 计算指标
    metrics = compute_metrics(y_true, y_pred, metrics_list)
    
    # 构建表格
    table = "\n" + "="*50 + "\n"
    table += "{:<15} {:>15}\n".format("指标", "值")
    table += "-"*50 + "\n"
    
    # 添加每个指标的值
    for metric_name, metric_value in metrics.items():
        if metric_name.lower() in ["mape", "smape", "wape"]:
            # 百分比指标
            table += "{:<15} {:>14.4f}%\n".format(metric_name.upper(), metric_value)
        else:
            # 其他指标
            table += "{:<15} {:>14.6f}\n".format(metric_name.upper(), metric_value)
    
    table += "="*50 + "\n"
    
    return table


def compare_models_metrics(models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    比较多个模型的评估指标
    
    Args:
        models_results: 字典，键为模型名称，值为该模型的评估指标字典
        
    Returns:
        pd.DataFrame: 包含所有模型评估指标的DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要安装pandas库: pip install pandas")
    
    # 转换为DataFrame
    df = pd.DataFrame(models_results).T
    
    # 添加最佳指标标记
    for metric in df.columns:
        if metric in ["mae", "rmse", "mse", "mape", "mre", "smape", "wape"]:
            # 这些指标越小越好
            best_value = df[metric].min()
            df[f"{metric}_best"] = df[metric] == best_value
        elif metric == "r2":
            # 这个指标越大越好
            best_value = df[metric].max()
            df[f"{metric}_best"] = df[metric] == best_value
    
    return df


def save_metrics_to_csv(metrics: Dict[str, float], filename: str, 
                        model_name: str = None, run_id: int = None):
    """
    保存指标到CSV文件
    
    Args:
        metrics: 评估指标字典
        filename: 输出CSV文件路径
        model_name: 模型名称（可选）
        run_id: 运行ID（可选）
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要安装pandas库: pip install pandas")
    
    # 创建DataFrame
    df = pd.DataFrame([metrics])
    
    # 添加模型名称和运行ID（如果提供）
    if model_name is not None:
        df["model"] = model_name
    if run_id is not None:
        df["run_id"] = run_id
    
    # 调整列顺序
    cols = df.columns.tolist()
    if model_name is not None:
        cols = ["model"] + [col for col in cols if col != "model"]
    if run_id is not None:
        cols = ["run_id"] + [col for col in cols if col != "run_id"]
    
    df = df[cols]
    
    # 检查文件是否存在
    import os
    if os.path.exists(filename):
        # 文件存在，追加数据
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # 文件不存在，创建新文件
        df.to_csv(filename, index=False)
    
    print(f"指标已保存到 {filename}")


def main():
    """
    测试指标计算函数
    """
    # 生成测试数据
    np.random.seed(42)
    y_true = np.random.rand(100, 20, 12)  # [batch_size, num_nodes, pred_len]
    y_pred = y_true + 0.1 * np.random.randn(100, 20, 12)  # 预测值略高于真实值
    
    # 计算所有指标
    print("计算所有指标...")
    metrics = calculate_all_metrics(y_true, y_pred)
    
    # 打印指标汇总表
    print(metrics_summary_table(y_true, y_pred))
    
    # 计算每个时间步的指标
    print("\n计算每个时间步的指标...")
    time_metrics = compute_time_horizon_metrics(y_true, y_pred, ["mae", "rmse"])
    print(f"时间步 MAE: {time_metrics['mae']}")
    print(f"时间步 RMSE: {time_metrics['rmse']}")
    
    # 计算每个节点的指标
    print("\n计算每个节点的指标...")
    node_metrics = compute_node_level_metrics(y_true, y_pred, ["mae", "rmse"])
    print(f"节点 MAE 平均值: {np.mean(node_metrics['mae'])}")
    print(f"节点 RMSE 平均值: {np.mean(node_metrics['rmse'])}")
    
    # 计算置信区间
    print("\n计算置信区间...")
    ci = compute_confidence_intervals(y_true, y_pred, metric_name="mae", n_bootstrap=100)
    print(f"MAE 95% 置信区间: [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}]")


if __name__ == "__main__":
    main()