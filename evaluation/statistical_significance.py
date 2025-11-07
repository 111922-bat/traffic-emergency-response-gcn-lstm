"""
统计显著性检验脚本

此脚本用于对模型评估结果进行统计分析，支持：
- 配对T检验 (Paired T-Test)
- Wilcoxon符号秩检验 (Wilcoxon Signed-Rank Test)
- 方差分析 (ANOVA)
- 事后检验 (如Tukey HSD)
- 效应量计算 (Cohen's d, eta-squared等)

使用方法：
    python statistical_significance.py --results_dir ../results/evaluations --output ../results/statistical_analysis
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import levene, pearsonr, kendalltau
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison, tukeyhsd
from datetime import datetime
import logging
import itertools
from typing import Dict, List, Tuple, Any, Optional, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'statistical_significance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('statistical_significance')

class StatisticalAnalyzer:
    """
    统计分析器类
    """
    
    def __init__(self, output_dir: str):
        """
        初始化统计分析器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        self.figures_dir = os.path.join(output_dir, 'figures')
        self.tables_dir = os.path.join(output_dir, 'tables')
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        
        # 数据存储
        self.results_data = None
        self.models = None
        self.metrics = None
    
    def load_results(self, results_dir: str, file_pattern: str = '*.json') -> bool:
        """
        从目录加载评估结果
        
        Args:
            results_dir: 结果目录路径
            file_pattern: 文件匹配模式
            
        Returns:
            是否成功加载
        """
        import glob
        
        logger.info(f"正在从 {results_dir} 加载结果文件")
        
        # 查找所有结果文件
        result_files = glob.glob(os.path.join(results_dir, file_pattern))
        
        if not result_files:
            logger.error(f"在 {results_dir} 中未找到结果文件")
            return False
        
        logger.info(f"找到 {len(result_files)} 个结果文件")
        
        # 加载所有结果
        all_results = []
        
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    
                # 提取基本信息
                file_name = os.path.basename(file_path)
                
                # 解析不同格式的结果文件
                if isinstance(results, dict):
                    # 检查是否有多次运行的结果
                    if 'runs' in results:
                        for run_idx, run in enumerate(results['runs']):
                            record = {
                                'file': file_name,
                                'run': run_idx + 1,
                                'model': results.get('model_name', 'unknown'),
                                'dataset': results.get('dataset', 'unknown')
                            }
                            # 添加所有指标
                            if isinstance(run, dict):
                                for key, value in run.items():
                                    if isinstance(value, (int, float)):
                                        record[key] = value
                            elif isinstance(run, float):
                                record['score'] = run
                            all_results.append(record)
                    else:
                        # 单条结果
                        record = {
                            'file': file_name,
                            'run': 1,
                            'model': results.get('model_name', 'unknown'),
                            'dataset': results.get('dataset', 'unknown')
                        }
                        # 添加所有指标
                        for key, value in results.items():
                            if isinstance(value, (int, float)) and key not in ['run', 'model', 'dataset']:
                                record[key] = value
                        all_results.append(record)
                elif isinstance(results, list):
                    # 结果列表
                    for idx, run in enumerate(results):
                        if isinstance(run, dict):
                            record = {
                                'file': file_name,
                                'run': idx + 1,
                                'model': run.get('model_name', 'unknown'),
                                'dataset': run.get('dataset', 'unknown')
                            }
                            # 添加所有指标
                            for key, value in run.items():
                                if isinstance(value, (int, float)) and key not in ['run', 'model', 'dataset']:
                                    record[key] = value
                            all_results.append(record)
            except Exception as e:
                logger.error(f"加载文件 {file_path} 时出错: {e}")
                continue
        
        if not all_results:
            logger.error("未能加载任何有效的结果数据")
            return False
        
        # 转换为DataFrame
        self.results_data = pd.DataFrame(all_results)
        
        # 提取模型和指标信息
        self.models = sorted(self.results_data['model'].unique())
        numeric_cols = self.results_data.select_dtypes(include=[np.number]).columns.tolist()
        self.metrics = [col for col in numeric_cols if col not in ['run']]
        
        logger.info(f"成功加载 {len(self.results_data)} 条记录")
        logger.info(f"模型: {', '.join(self.models)}")
        logger.info(f"指标: {', '.join(self.metrics)}")
        
        # 保存加载的数据
        self.results_data.to_csv(os.path.join(self.tables_dir, 'loaded_results.csv'), index=False)
        
        return True
    
    def load_custom_results(self, custom_data: List[Dict[str, Any]]) -> bool:
        """
        加载自定义结果数据
        
        Args:
            custom_data: 自定义结果数据列表
            
        Returns:
            是否成功加载
        """
        try:
            self.results_data = pd.DataFrame(custom_data)
            
            # 确保必要的列存在
            required_cols = ['model', 'run']
            for col in required_cols:
                if col not in self.results_data.columns:
                    logger.error(f"缺少必要的列: {col}")
                    return False
            
            # 提取模型和指标信息
            self.models = sorted(self.results_data['model'].unique())
            numeric_cols = self.results_data.select_dtypes(include=[np.number]).columns.tolist()
            self.metrics = [col for col in numeric_cols if col not in ['run']]
            
            logger.info(f"成功加载 {len(self.results_data)} 条自定义记录")
            logger.info(f"模型: {', '.join(self.models)}")
            logger.info(f"指标: {', '.join(self.metrics)}")
            
            return True
        except Exception as e:
            logger.error(f"加载自定义数据时出错: {e}")
            return False
    
    def check_normality(self, metric: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        检查数据的正态分布性
        
        Args:
            metric: 要检查的指标
            alpha: 显著性水平
            
        Returns:
            正态性检验结果
        """
        if metric not in self.metrics:
            logger.error(f"指标 {metric} 不存在")
            return {}
        
        results = {}
        
        # 对每个模型进行检验
        for model in self.models:
            model_data = self.results_data[self.results_data['model'] == model][metric].dropna().values
            
            if len(model_data) < 3:
                logger.warning(f"模型 {model} 的样本数量不足，无法进行正态性检验")
                results[model] = {
                    'valid': False,
                    'message': '样本数量不足'
                }
                continue
            
            # 执行Shapiro-Wilk检验（适用于小样本）
            if len(model_data) <= 5000:
                stat, p_value = stats.shapiro(model_data)
                test_name = 'Shapiro-Wilk'
            else:
                # 大样本使用D'Agostino's K^2检验
                stat, p_value = stats.normaltest(model_data)
                test_name = "D'Agostino's K^2"
            
            # 判断是否服从正态分布
            is_normal = p_value > alpha
            
            results[model] = {
                'valid': True,
                'test_name': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': is_normal,
                'sample_size': len(model_data),
                'alpha': alpha
            }
            
            logger.info(f"模型 {model} ({metric}) - {test_name} 检验: 统计量={stat:.4f}, p值={p_value:.4f}, "
                      f"{'服从' if is_normal else '不服从'}正态分布")
        
        # 绘制直方图和Q-Q图
        self._plot_normality_checks(metric, results)
        
        return results
    
    def check_homogeneity(self, metric: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        检查方差齐性
        
        Args:
            metric: 要检查的指标
            alpha: 显著性水平
            
        Returns:
            方差齐性检验结果
        """
        if metric not in self.metrics:
            logger.error(f"指标 {metric} 不存在")
            return {}
        
        # 收集所有模型的数据
        model_data_list = []
        valid_models = []
        
        for model in self.models:
            model_data = self.results_data[self.results_data['model'] == model][metric].dropna().values
            if len(model_data) > 0:
                model_data_list.append(model_data)
                valid_models.append(model)
        
        if len(model_data_list) < 2:
            logger.error("有效的模型数量不足，无法进行方差齐性检验")
            return {}
        
        # 执行Levene检验
        stat, p_value = levene(*model_data_list)
        
        # 判断方差是否齐性
        homogeneity = p_value > alpha
        
        result = {
            'test_name': 'Levene',
            'statistic': float(stat),
            'p_value': float(p_value),
            'homogeneity': homogeneity,
            'models': valid_models,
            'alpha': alpha
        }
        
        logger.info(f"方差齐性检验 ({metric}) - Levene检验: 统计量={stat:.4f}, p值={p_value:.4f}, "
                  f"{'方差齐性' if homogeneity else '方差不齐'}")
        
        # 绘制箱线图
        self._plot_boxplot(metric)
        
        return result
    
    def paired_ttest(self, model1: str, model2: str, metric: str, alpha: float = 0.05, 
                    check_normality: bool = True) -> Dict[str, Any]:
        """
        配对T检验
        
        Args:
            model1: 第一个模型
            model2: 第二个模型
            metric: 要比较的指标
            alpha: 显著性水平
            check_normality: 是否先检查正态性
            
        Returns:
            T检验结果
        """
        if model1 not in self.models or model2 not in self.models:
            logger.error(f"模型 {model1} 或 {model2} 不存在")
            return {}
        
        if metric not in self.metrics:
            logger.error(f"指标 {metric} 不存在")
            return {}
        
        # 获取两个模型的数据
        model1_data = self.results_data[self.results_data['model'] == model1][metric].dropna().values
        model2_data = self.results_data[self.results_data['model'] == model2][metric].dropna().values
        
        # 确保样本大小相同（配对T检验要求）
        min_len = min(len(model1_data), len(model2_data))
        model1_data = model1_data[:min_len]
        model2_data = model2_data[:min_len]
        
        if min_len < 2:
            logger.error("样本数量不足，无法进行配对T检验")
            return {}
        
        # 检查正态性（如果需要）
        normality_result = None
        if check_normality:
            # 对差值进行正态性检验
            differences = model1_data - model2_data
            
            if len(differences) <= 5000:
                norm_stat, norm_p_value = stats.shapiro(differences)
                test_name = 'Shapiro-Wilk'
            else:
                norm_stat, norm_p_value = stats.normaltest(differences)
                test_name = "D'Agostino's K^2"
            
            is_normal = norm_p_value > alpha
            
            normality_result = {
                'test_name': test_name,
                'statistic': float(norm_stat),
                'p_value': float(norm_p_value),
                'is_normal': is_normal
            }
            
            logger.info(f"差值正态性检验 - {test_name}: 统计量={norm_stat:.4f}, p值={norm_p_value:.4f}, "
                      f"{'服从' if is_normal else '不服从'}正态分布")
        
        # 执行配对T检验
        try:
            stat, p_value = stats.ttest_rel(model1_data, model2_data)
            
            # 计算效应量 (Cohen's d)
            mean_diff = np.mean(model1_data - model2_data)
            std_diff = np.std(model1_data - model2_data, ddof=1)
            cohens_d = mean_diff / std_diff
            
            # 置信区间
            confidence_level = 1 - alpha
            df = len(model1_data) - 1
            sem = stats.sem(model1_data - model2_data)
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
            ci_lower = mean_diff - t_critical * sem
            ci_upper = mean_diff + t_critical * sem
            
            # 效应量解释
            effect_size_interpretation = self._interpret_cohens_d(abs(cohens_d))
            
            result = {
                'test_name': 'Paired T-Test',
                'model1': model1,
                'model2': model2,
                'metric': metric,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha,
                'sample_size': min_len,
                'mean_model1': float(np.mean(model1_data)),
                'mean_model2': float(np.mean(model2_data)),
                'mean_difference': float(mean_diff),
                'cohens_d': float(cohens_d),
                'effect_size_interpretation': effect_size_interpretation,
                'confidence_interval': {
                    'level': confidence_level,
                    'lower': float(ci_lower),
                    'upper': float(ci_upper)
                },
                'normality_check': normality_result
            }
            
            # 记录结果
            significance = '显著' if p_value < alpha else '不显著'
            direction = '更好' if (mean_diff < 0 and 'mae' in metric.lower()) or \
                                 (mean_diff > 0 and 'r2' in metric.lower()) else '更差'
            
            logger.info(f"配对T检验 ({model1} vs {model2}, {metric}): "
                      f"统计量={stat:.4f}, p值={p_value:.4f}, {significance}差异")
            logger.info(f"效应量 (Cohen's d): {cohens_d:.4f} ({effect_size_interpretation})")
            logger.info(f"{model1} 平均={np.mean(model1_data):.4f}, {model2} 平均={np.mean(model2_data):.4f}")
            logger.info(f"95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # 绘制比较图
            self._plot_model_comparison(model1, model2, metric, result)
            
            return result
            
        except Exception as e:
            logger.error(f"执行配对T检验时出错: {e}")
            return {}
    
    def wilcoxon_test(self, model1: str, model2: str, metric: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Wilcoxon符号秩检验（非参数检验）
        
        Args:
            model1: 第一个模型
            model2: 第二个模型
            metric: 要比较的指标
            alpha: 显著性水平
            
        Returns:
            Wilcoxon检验结果
        """
        if model1 not in self.models or model2 not in self.models:
            logger.error(f"模型 {model1} 或 {model2} 不存在")
            return {}
        
        if metric not in self.metrics:
            logger.error(f"指标 {metric} 不存在")
            return {}
        
        # 获取两个模型的数据
        model1_data = self.results_data[self.results_data['model'] == model1][metric].dropna().values
        model2_data = self.results_data[self.results_data['model'] == model2][metric].dropna().values
        
        # 确保样本大小相同
        min_len = min(len(model1_data), len(model2_data))
        model1_data = model1_data[:min_len]
        model2_data = model2_data[:min_len]
        
        if min_len < 2:
            logger.error("样本数量不足，无法进行Wilcoxon检验")
            return {}
        
        # 计算差值
        differences = model1_data - model2_data
        
        # 移除差值为0的情况
        non_zero_indices = differences != 0
        if not np.any(non_zero_indices):
            logger.error("所有差值为0，无法进行Wilcoxon检验")
            return {}
        
        model1_data = model1_data[non_zero_indices]
        model2_data = model2_data[non_zero_indices]
        differences = differences[non_zero_indices]
        
        try:
            # 执行Wilcoxon符号秩检验
            stat, p_value = stats.wilcoxon(model1_data, model2_data)
            
            # 计算效应量 (r = z / sqrt(n))
            z_stat = stats.norm.ppf(p_value / 2)
            r_effect = abs(z_stat) / np.sqrt(len(differences))
            
            # 效应量解释
            effect_size_interpretation = self._interpret_effect_size_r(r_effect)
            
            result = {
                'test_name': 'Wilcoxon Signed-Rank Test',
                'model1': model1,
                'model2': model2,
                'metric': metric,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha,
                'sample_size': len(model1_data),
                'mean_model1': float(np.mean(model1_data)),
                'mean_model2': float(np.mean(model2_data)),
                'median_difference': float(np.median(differences)),
                'effect_size_r': float(r_effect),
                'effect_size_interpretation': effect_size_interpretation
            }
            
            # 记录结果
            significance = '显著' if p_value < alpha else '不显著'
            
            logger.info(f"Wilcoxon检验 ({model1} vs {model2}, {metric}): "
                      f"统计量={stat:.4f}, p值={p_value:.4f}, {significance}差异")
            logger.info(f"效应量 (r): {r_effect:.4f} ({effect_size_interpretation})")
            logger.info(f"{model1} 平均={np.mean(model1_data):.4f}, {model2} 平均={np.mean(model2_data):.4f}")
            
            # 绘制比较图
            self._plot_model_comparison(model1, model2, metric, result)
            
            return result
            
        except Exception as e:
            logger.error(f"执行Wilcoxon检验时出错: {e}")
            return {}
    
    def anova_test(self, metric: str, alpha: float = 0.05, check_assumptions: bool = True) -> Dict[str, Any]:
        """
        单因素方差分析
        
        Args:
            metric: 要比较的指标
            alpha: 显著性水平
            check_assumptions: 是否检查假设条件
            
        Returns:
            ANOVA结果
        """
        if metric not in self.metrics:
            logger.error(f"指标 {metric} 不存在")
            return {}
        
        # 准备数据
        data_for_anova = self.results_data[['model', metric]].dropna()
        
        if len(data_for_anova) < len(self.models):
            logger.error("数据不足，无法进行方差分析")
            return {}
        
        # 检查假设条件
        assumptions = {}
        
        if check_assumptions:
            # 检查正态性
            normality_results = self.check_normality(metric, alpha)
            all_normal = all(result.get('is_normal', False) for result in normality_results.values())
            
            # 检查方差齐性
            homogeneity_result = self.check_homogeneity(metric, alpha)
            has_homogeneity = homogeneity_result.get('homogeneity', False)
            
            assumptions = {
                'normality': normality_results,
                'all_normal': all_normal,
                'homogeneity': homogeneity_result,
                'has_homogeneity': has_homogeneity
            }
            
            logger.info(f"ANOVA假设检查 - 所有模型正态性: {all_normal}")
            logger.info(f"ANOVA假设检查 - 方差齐性: {has_homogeneity}")
        
        try:
            # 使用statsmodels进行ANOVA
            model = ols(f'{metric} ~ C(model)', data=data_for_anova).fit()
            anova_table = anova_lm(model, typ=2)
            
            # 提取结果
            f_stat = float(anova_table.loc['C(model)', 'F'])
            p_value = float(anova_table.loc['C(model)', 'PR(>F)'])
            df_model = int(anova_table.loc['C(model)', 'df'])
            df_residual = int(anova_table.loc['Residual', 'df'])
            
            # 计算效应量 (eta-squared)
            ss_model = float(anova_table.loc['C(model)', 'sum_sq'])
            ss_total = ss_model + float(anova_table.loc['Residual', 'sum_sq'])
            eta_squared = ss_model / ss_total
            
            # 效应量解释
            effect_size_interpretation = self._interpret_eta_squared(eta_squared)
            
            result = {
                'test_name': 'One-Way ANOVA',
                'metric': metric,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'alpha': alpha,
                'degrees_of_freedom': {
                    'model': df_model,
                    'residual': df_residual
                },
                'eta_squared': eta_squared,
                'effect_size_interpretation': effect_size_interpretation,
                'models_compared': self.models,
                'assumptions': assumptions,
                'anova_table': anova_table.to_dict()
            }
            
            # 记录结果
            significance = '显著' if p_value < alpha else '不显著'
            
            logger.info(f"ANOVA检验 ({metric}): F={f_stat:.4f}, p值={p_value:.4f}, {significance}差异")
            logger.info(f"效应量 (eta-squared): {eta_squared:.4f} ({effect_size_interpretation})")
            
            # 保存ANOVA表格
            anova_table.to_csv(os.path.join(self.tables_dir, f'anova_table_{metric}.csv'))
            
            # 如果差异显著，执行事后检验
            if p_value < alpha:
                tukey_result = self.tukey_hsd(metric, alpha)
                result['post_hoc'] = tukey_result
            
            return result
            
        except Exception as e:
            logger.error(f"执行ANOVA检验时出错: {e}")
            return {}
    
    def tukey_hsd(self, metric: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Tukey HSD事后检验
        
        Args:
            metric: 要比较的指标
            alpha: 显著性水平
            
        Returns:
            Tukey HSD检验结果
        """
        if metric not in self.metrics:
            logger.error(f"指标 {metric} 不存在")
            return {}
        
        # 准备数据
        data_for_tukey = self.results_data[['model', metric]].dropna()
        
        try:
            # 执行Tukey HSD检验
            mc = MultiComparison(data_for_tukey[metric], data_for_tukey['model'])
            result = mc.tukeyhsd(alpha=alpha)
            
            # 转换结果为DataFrame
            tukey_table = pd.DataFrame({
                'group1': result.groupsunique[result.groupindices[0]],
                'group2': result.groupsunique[result.groupindices[1]],
                'meandiff': result.meandiffs,
                'p-adj': result.pvalues,
                'lower': result.confint[:, 0],
                'upper': result.confint[:, 1],
                'reject': result.reject
            })
            
            # 保存结果
            tukey_table.to_csv(os.path.join(self.tables_dir, f'tukey_hsd_{metric}.csv'), index=False)
            
            # 记录显著差异的配对
            significant_pairs = tukey_table[tukey_table['reject']]
            logger.info(f"Tukey HSD检验 ({metric}) 发现 {len(significant_pairs)} 对显著差异")
            
            for _, row in significant_pairs.iterrows():
                direction = '更好' if (row['meandiff'] < 0 and 'mae' in metric.lower()) or \
                                     (row['meandiff'] > 0 and 'r2' in metric.lower()) else '更差'
                logger.info(f"  {row['group1']} vs {row['group2']}: 差异={row['meandiff']:.4f}, "
                          f"p值={row['p-adj']:.4f}, {row['group1']} {direction}")
            
            # 绘制结果图
            self._plot_tukey_results(result, metric)
            
            return {
                'test_name': 'Tukey HSD',
                'metric': metric,
                'alpha': alpha,
                'significant_pairs': significant_pairs.to_dict('records'),
                'summary': str(result),
                'table': tukey_table.to_dict()
            }
            
        except Exception as e:
            logger.error(f"执行Tukey HSD检验时出错: {e}")
            return {}
    
    def friedman_test(self, metric: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Friedman检验（非参数方差分析）
        
        Args:
            metric: 要比较的指标
            alpha: 显著性水平
            
        Returns:
            Friedman检验结果
        """
        if metric not in self.metrics:
            logger.error(f"指标 {metric} 不存在")
            return {}
        
        # 准备数据：将数据重塑为适合Friedman检验的格式
        # 按run分组，每个run有多个模型的结果
        pivot_data = self.results_data.pivot(index='run', columns='model', values=metric)
        
        if pivot_data.isnull().values.any():
            logger.warning("数据中存在缺失值，将被排除")
            # 只保留没有缺失值的行
            pivot_data = pivot_data.dropna()
        
        if len(pivot_data) < 2:
            logger.error("有效样本数量不足，无法进行Friedman检验")
            return {}
        
        try:
            # 执行Friedman检验
            stat, p_value = stats.friedmanchisquare(*[pivot_data[model] for model in pivot_data.columns])
            
            # 计算效应量 (Kendall's W)
            n = len(pivot_data)  # 块数
            k = len(pivot_data.columns)  # 处理数
            kendalls_w = stat / (n * (k - 1))
            
            # 效应量解释
            effect_size_interpretation = self._interpret_kendalls_w(kendalls_w)
            
            result = {
                'test_name': 'Friedman Test',
                'metric': metric,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha,
                'blocks': n,
                'treatments': k,
                'kendalls_w': float(kendalls_w),
                'effect_size_interpretation': effect_size_interpretation,
                'models_compared': list(pivot_data.columns)
            }
            
            # 记录结果
            significance = '显著' if p_value < alpha else '不显著'
            
            logger.info(f"Friedman检验 ({metric}): 统计量={stat:.4f}, p值={p_value:.4f}, {significance}差异")
            logger.info(f"效应量 (Kendall's W): {kendalls_w:.4f} ({effect_size_interpretation})")
            
            # 如果差异显著，可以执行Conover的事后检验
            if p_value < alpha:
                logger.info("建议执行Conover的事后检验进行两两比较")
            
            return result
            
        except Exception as e:
            logger.error(f"执行Friedman检验时出错: {e}")
            return {}
    
    def run_all_paired_tests(self, metric: str, alpha: float = 0.05, 
                           adjust_pvalue: bool = True) -> Dict[str, Any]:
        """
        对所有模型对执行配对T检验
        
        Args:
            metric: 要比较的指标
            alpha: 显著性水平
            adjust_pvalue: 是否进行p值调整（多重比较校正）
            
        Returns:
            所有配对检验结果
        """
        if metric not in self.metrics:
            logger.error(f"指标 {metric} 不存在")
            return {}
        
        # 生成所有模型对
        model_pairs = list(itertools.combinations(self.models, 2))
        
        results = {
            'metric': metric,
            'alpha': alpha,
            'adjust_pvalue': adjust_pvalue,
            'pairs': {},
            'summary': {}
        }
        
        # 存储所有p值用于多重比较校正
        p_values = []
        pair_names = []
        
        # 对每对模型执行检验
        for model1, model2 in model_pairs:
            pair_key = f"{model1}_vs_{model2}"
            logger.info(f"执行配对T检验: {model1} vs {model2} ({metric})")
            
            # 先检查正态性
            normality_results = self.check_normality(metric, alpha)
            model1_normal = normality_results.get(model1, {}).get('is_normal', False)
            model2_normal = normality_results.get(model2, {}).get('is_normal', False)
            
            if model1_normal and model2_normal:
                # 数据正态，使用配对T检验
                test_result = self.paired_ttest(model1, model2, metric, alpha, check_normality=False)
            else:
                # 数据非正态，使用Wilcoxon检验
                test_result = self.wilcoxon_test(model1, model2, metric, alpha)
            
            if test_result:
                results['pairs'][pair_key] = test_result
                p_values.append(test_result['p_value'])
                pair_names.append(pair_key)
        
        # 多重比较校正
        if adjust_pvalue and p_values:
            # 使用Bonferroni校正
            bonferroni_alpha = alpha / len(p_values)
            results['bonferroni_alpha'] = bonferroni_alpha
            
            # 使用Holm-Bonferroni校正（比Bonferroni更强大）
            sorted_indices = np.argsort(p_values)
            sorted_p_values = np.array(p_values)[sorted_indices]
            sorted_pairs = np.array(pair_names)[sorted_indices]
            
            holm_adjusted = []
            for i, (p_val, pair) in enumerate(zip(sorted_p_values, sorted_pairs)):
                adjusted_p = p_val * (len(p_values) - i)
                holm_adjusted.append(min(adjusted_p, 1.0))
                results['pairs'][pair]['holm_adjusted_p'] = min(adjusted_p, 1.0)
                results['pairs'][pair]['significant_after_holm'] = adjusted_p < alpha
            
            results['holm_adjusted_pvalues'] = dict(zip(sorted_pairs, holm_adjusted))
            logger.info(f"多重比较校正 - Bonferroni校正后的α: {bonferroni_alpha:.6f}")
        
        # 生成汇总信息
        significant_pairs = []
        for pair_key, pair_result in results['pairs'].items():
            if adjust_pvalue:
                is_significant = pair_result.get('significant_after_holm', False)
            else:
                is_significant = pair_result.get('significant', False)
            
            if is_significant:
                significant_pairs.append({
                    'pair': pair_key,
                    'p_value': pair_result.get('p_value'),
                    'adjusted_p_value': pair_result.get('holm_adjusted_p'),
                    'effect_size': pair_result.get('cohens_d') or pair_result.get('effect_size_r'),
                    'effect_interpretation': pair_result.get('effect_size_interpretation')
                })
        
        results['summary']['total_pairs'] = len(model_pairs)
        results['summary']['significant_pairs'] = len(significant_pairs)
        results['summary']['significant_details'] = significant_pairs
        
        # 生成热图显示所有比较结果
        self._plot_comparison_heatmap(results, metric)
        
        # 保存所有结果
        with open(os.path.join(self.tables_dir, f'all_pairwise_tests_{metric}.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"所有配对检验完成。发现 {len(significant_pairs)} 对显著差异")
        
        return results
    
    def compute_confidence_intervals(self, metric: str, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        计算每个模型指标的置信区间
        
        Args:
            metric: 要计算的指标
            confidence_level: 置信水平
            
        Returns:
            置信区间结果
        """
        if metric not in self.metrics:
            logger.error(f"指标 {metric} 不存在")
            return {}
        
        results = {
            'metric': metric,
            'confidence_level': confidence_level,
            'models': {}
        }
        
        for model in self.models:
            model_data = self.results_data[self.results_data['model'] == model][metric].dropna().values
            
            if len(model_data) < 2:
                logger.warning(f"模型 {model} 的样本数量不足，无法计算置信区间")
                results['models'][model] = {
                    'valid': False,
                    'message': '样本数量不足'
                }
                continue
            
            # 计算均值和标准差
            mean_val = np.mean(model_data)
            std_val = np.std(model_data, ddof=1)
            n = len(model_data)
            
            # 计算标准误
            sem = std_val / np.sqrt(n)
            
            # 计算置信区间
            alpha = 1 - confidence_level
            
            # 检查正态性决定使用t分布还是z分布
            if len(model_data) <= 30:
                # 小样本使用t分布
                critical_value = stats.t.ppf(1 - alpha/2, df=n-1)
                distribution = 't'
            else:
                # 大样本使用z分布
                critical_value = stats.norm.ppf(1 - alpha/2)
                distribution = 'z'
            
            margin_of_error = critical_value * sem
            ci_lower = mean_val - margin_of_error
            ci_upper = mean_val + margin_of_error
            
            results['models'][model] = {
                'valid': True,
                'mean': float(mean_val),
                'std': float(std_val),
                'sample_size': n,
                'distribution': distribution,
                'critical_value': float(critical_value),
                'margin_of_error': float(margin_of_error),
                'confidence_interval': {
                    'lower': float(ci_lower),
                    'upper': float(ci_upper)
                },
                'relative_error': float(margin_of_error / mean_val * 100) if mean_val != 0 else float('inf')
            }
            
            logger.info(f"模型 {model} ({metric}) - 均值: {mean_val:.4f}, "
                      f"{int(confidence_level*100)}%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}] ({distribution}分布)")
        
        # 绘制置信区间图
        self._plot_confidence_intervals(results, metric)
        
        # 保存结果
        with open(os.path.join(self.tables_dir, f'confidence_intervals_{metric}.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return results
    
    def generate_detailed_report(self, output_file: str = None) -> Dict[str, Any]:
        """
        生成详细的统计分析报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            报告内容
        """
        report = {
            'title': '统计显著性分析报告',
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'metrics_analysis': {},
            'visualizations': {},
            'tables': {}
        }
        
        # 基本统计信息
        report['summary'] = {
            'total_records': len(self.results_data),
            'models': self.models,
            'metrics': self.metrics,
            'runs_per_model': {}
        }
        
        # 计算每个模型的运行次数
        for model in self.models:
            report['summary']['runs_per_model'][model] = len(
                self.results_data[self.results_data['model'] == model]
            )
        
        # 对每个指标进行分析
        for metric in self.metrics:
            logger.info(f"开始分析指标: {metric}")
            
            metric_analysis = {
                'basic_statistics': {},
                'normality_tests': {},
                'homogeneity_test': {},
                'confidence_intervals': {},
                'anova_result': {},
                'pairwise_comparisons': {}
            }
            
            # 基本统计
            for model in self.models:
                model_data = self.results_data[self.results_data['model'] == model][metric].dropna()
                metric_analysis['basic_statistics'][model] = {
                    'mean': float(model_data.mean()),
                    'std': float(model_data.std()),
                    'median': float(model_data.median()),
                    'min': float(model_data.min()),
                    'max': float(model_data.max()),
                    'q25': float(model_data.quantile(0.25)),
                    'q75': float(model_data.quantile(0.75)),
                    'count': int(len(model_data))
                }
            
            # 正态性检验
            metric_analysis['normality_tests'] = self.check_normality(metric)
            
            # 方差齐性检验
            metric_analysis['homogeneity_test'] = self.check_homogeneity(metric)
            
            # 置信区间
            metric_analysis['confidence_intervals'] = self.compute_confidence_intervals(metric)
            
            # ANOVA检验
            anova_result = self.anova_test(metric)
            metric_analysis['anova_result'] = anova_result
            
            # 两两比较
            pairwise_results = self.run_all_paired_tests(metric)
            metric_analysis['pairwise_comparisons'] = pairwise_results
            
            report['metrics_analysis'][metric] = metric_analysis
        
        # 生成关键发现
        key_findings = []
        
        for metric, analysis in report['metrics_analysis'].items():
            # 检查ANOVA结果
            if analysis['anova_result'].get('significant', False):
                key_findings.append(
                    f"ANOVA分析显示，在 {metric} 指标上，不同模型之间存在显著差异 (p = {analysis['anova_result'].get('p_value', 'N/A'):.4f})"
                )
            
            # 检查两两比较的显著结果
            significant_count = analysis['pairwise_comparisons'].get('summary', {}).get('significant_pairs', 0)
            if significant_count > 0:
                key_findings.append(
                    f"在 {metric} 指标上，共发现 {significant_count} 对模型之间存在统计学显著差异（经过多重比较校正）"
                )
            
            # 识别最佳模型
            if analysis['basic_statistics']:
                # 对于误差指标（MAE, RMSE, MAPE），值越小越好
                if any(keyword in metric.lower() for keyword in ['mae', 'rmse', 'mape']):
                    best_model = min(analysis['basic_statistics'].items(), key=lambda x: x[1]['mean'])
                # 对于R²等指标，值越大越好
                elif 'r2' in metric.lower():
                    best_model = max(analysis['basic_statistics'].items(), key=lambda x: x[1]['mean'])
                else:
                    continue
                
                best_model_name, best_stats = best_model
                ci = analysis['confidence_intervals'].get('models', {}).get(best_model_name, {})
                
                if ci and ci.get('valid', False):
                    ci_lower = ci['confidence_interval']['lower']
                    ci_upper = ci['confidence_interval']['upper']
                    key_findings.append(
                        f"模型 {best_model_name} 在 {metric} 指标上表现最佳，平均={best_stats['mean']:.4f}，"\
                        f"95%置信区间=[{ci_lower:.4f}, {ci_upper:.4f}]"
                    )
        
        report['key_findings'] = key_findings
        
        # 保存报告
        if not output_file:
            output_file = os.path.join(self.output_dir, 'statistical_analysis_report.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成HTML报告
        self._generate_html_report(report)
        
        logger.info(f"详细统计分析报告已生成: {output_file}")
        
        return report
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """
        解释Cohen's d效应量
        """
        if cohens_d < 0.2:
            return "微小效应"
        elif cohens_d < 0.5:
            return "小效应"
        elif cohens_d < 0.8:
            return "中等效应"
        else:
            return "大效应"
    
    def _interpret_effect_size_r(self, r_value: float) -> str:
        """
        解释相关系数r效应量
        """
        r_abs = abs(r_value)
        if r_abs < 0.1:
            return "微小效应"
        elif r_abs < 0.3:
            return "小效应"
        elif r_abs < 0.5:
            return "中等效应"
        elif r_abs < 0.7:
            return "大效应"
        elif r_abs < 0.9:
            return "很大效应"
        else:
            return "极大效应"
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """
        解释eta-squared效应量
        """
        if eta_squared < 0.01:
            return "微小效应"
        elif eta_squared < 0.06:
            return "小效应"
        elif eta_squared < 0.14:
            return "中等效应"
        else:
            return "大效应"
    
    def _interpret_kendalls_w(self, kendalls_w: float) -> str:
        """
        解释Kendall's W效应量
        """
        if kendalls_w < 0.1:
            return "微小效应"
        elif kendalls_w < 0.3:
            return "小效应"
        elif kendalls_w < 0.5:
            return "中等效应"
        else:
            return "大效应"
    
    def _plot_normality_checks(self, metric: str, normality_results: Dict[str, Any]):
        """
        绘制正态性检验可视化
        """
        try:
            # 每个模型绘制直方图和Q-Q图
            for model, result in normality_results.items():
                if not result.get('valid', False):
                    continue
                
                model_data = self.results_data[self.results_data['model'] == model][metric].dropna().values
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # 直方图
                ax1.hist(model_data, bins=20, alpha=0.7, color='skyblue')
                ax1.set_title(f'{model} - {metric} 直方图')
                ax1.set_xlabel(metric)
                ax1.set_ylabel('频率')
                ax1.grid(True, alpha=0.3)
                
                # 添加正态分布曲线
                x = np.linspace(min(model_data), max(model_data), 100)
                mean, std = stats.norm.fit(model_data)
                pdf = stats.norm.pdf(x, mean, std)
                ax1.plot(x, pdf * len(model_data) * (max(model_data) - min(model_data)) / 20, 
                        'r-', linewidth=2, label=f'正态分布\nμ={mean:.2f}, σ={std:.2f}')
                ax1.legend()
                
                # Q-Q图
                stats.probplot(model_data, dist="norm", plot=ax2)
                ax2.set_title(f'{model} - {metric} Q-Q图')
                ax2.grid(True, alpha=0.3)
                
                # 添加正态性检验结果
                test_name = result.get('test_name', 'Unknown')
                p_value = result.get('p_value', 'N/A')
                is_normal = '是' if result.get('is_normal', False) else '否'
                
                plt.figtext(0.5, 0.01, 
                           f'{test_name}检验: p值={p_value:.4f}, 服从正态分布: {is_normal}', 
                           ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.3})
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(self.figures_dir, f'normality_{model}_{metric}.png'), dpi=300)
                plt.close()
        except Exception as e:
            logger.error(f"绘制正态性检验图时出错: {e}")
    
    def _plot_boxplot(self, metric: str):
        """
        绘制箱线图
        """
        try:
            plt.figure(figsize=(10, 6))
            
            sns.boxplot(x='model', y=metric, data=self.results_data)
            
            # 添加散点图显示实际数据点
            sns.stripplot(x='model', y=metric, data=self.results_data, 
                         color='black', size=3, jitter=True, alpha=0.3)
            
            plt.title(f'不同模型的{metric}指标分布')
            plt.xlabel('模型')
            plt.ylabel(metric)
            plt.grid(True, axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.figures_dir, f'boxplot_{metric}.png'), dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"绘制箱线图时出错: {e}")
    
    def _plot_model_comparison(self, model1: str, model2: str, metric: str, result: Dict[str, Any]):
        """
        绘制两个模型的比较图
        """
        try:
            # 获取数据
            model1_data = self.results_data[self.results_data['model'] == model1][metric].dropna().values
            model2_data = self.results_data[self.results_data['model'] == model2][metric].dropna().values
            
            # 确保样本大小相同
            min_len = min(len(model1_data), len(model2_data))
            model1_data = model1_data[:min_len]
            model2_data = model2_data[:min_len]
            
            # 准备数据用于绘图
            df_compare = pd.DataFrame({
                'Run': range(1, min_len + 1),
                model1: model1_data,
                model2: model2_data
            })
            
            # 转换为长格式
            df_melt = pd.melt(df_compare, id_vars=['Run'], value_vars=[model1, model2],
                             var_name='Model', value_name=metric)
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. 线图显示每次运行的对比
            sns.lineplot(x='Run', y=metric, hue='Model', data=df_melt, ax=ax1, markers=True)
            ax1.set_title(f'{model1} vs {model2} - 每次运行的{metric}对比')
            ax1.set_xlabel('运行次数')
            ax1.set_ylabel(metric)
            ax1.grid(True, alpha=0.3)
            
            # 2. 散点图显示两个模型的相关性
            ax2.scatter(model1_data, model2_data, alpha=0.6)
            
            # 添加参考线 (y=x)
            min_val = min(min(model1_data), min(model2_data))
            max_val = max(max(model1_data), max(model2_data))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
            
            # 计算相关系数
            if len(model1_data) > 1:
                corr, _ = pearsonr(model1_data, model2_data)
                ax2.annotate(f'相关系数: {corr:.3f}', xy=(0.05, 0.95), 
                            xycoords='axes fraction', fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.8))
            
            ax2.set_title(f'{model1} vs {model2} - {metric}散点图')
            ax2.set_xlabel(model1)
            ax2.set_ylabel(model2)
            ax2.grid(True, alpha=0.3)
            
            # 添加统计检验结果
            test_name = result.get('test_name', 'Test')
            p_value = result.get('p_value', 'N/A')
            significance = '显著' if result.get('significant', False) else '不显著'
            effect_size = result.get('cohens_d') or result.get('effect_size_r')
            effect_interpretation = result.get('effect_size_interpretation', '')
            
            plt.figtext(0.5, 0.01, 
                       f'{test_name}: p值={p_value:.4f}, 差异{significance}, '\
                       f'效应量={effect_size:.4f} ({effect_interpretation})', 
                       ha="center", fontsize=10, bbox={"facecolor":"lightgreen", "alpha":0.3})
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(self.figures_dir, f'comparison_{model1}_vs_{model2}_{metric}.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制模型比较图时出错: {e}")
    
    def _plot_tukey_results(self, tukey_result, metric: str):
        """
        绘制Tukey HSD检验结果
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 提取数据
            groups = tukey_result.groupsunique
            meandiffs = tukey_result.meandiffs
            lower = tukey_result.confint[:, 0]
            upper = tukey_result.confint[:, 1]
            reject = tukey_result.reject
            
            # 创建配对标签
            pair_labels = []
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    pair_labels.append(f"{groups[i]}-{groups[j]}")
            
            # 绘制置信区间
            y_pos = range(len(meandiffs))
            
            # 绘制非显著差异的区间
            for i in y_pos:
                if not reject[i]:
                    ax.plot([lower[i], upper[i]], [y_pos[i], y_pos[i]], 'b-', linewidth=2)
                    ax.plot(meandiffs[i], y_pos[i], 'bo')
            
            # 绘制显著差异的区间（红色）
            for i in y_pos:
                if reject[i]:
                    ax.plot([lower[i], upper[i]], [y_pos[i], y_pos[i]], 'r-', linewidth=2)
                    ax.plot(meandiffs[i], y_pos[i], 'ro')
            
            # 添加零参考线
            ax.axvline(x=0, color='k', linestyle='--')
            
            # 设置标签
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_labels)
            ax.set_xlabel('均值差异')
            ax.set_title(f'Tukey HSD检验结果 ({metric})')
            ax.grid(True, axis='x', alpha=0.3)
            
            # 添加图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='b', lw=2, marker='o', label='非显著差异'),
                Line2D([0], [0], color='r', lw=2, marker='o', label='显著差异')
            ]
            ax.legend(handles=legend_elements, loc='best')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f'tukey_results_{metric}.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制Tukey HSD结果图时出错: {e}")
    
    def _plot_comparison_heatmap(self, pairwise_results: Dict[str, Any], metric: str):
        """
        绘制两两比较结果的热图
        """
        try:
            # 准备数据
            models = self.models
            n = len(models)
            
            # 创建矩阵存储p值和显著性
            p_value_matrix = np.eye(n) * np.nan  # 对角线为NaN
            significance_matrix = np.zeros((n, n), dtype=int)  # 0:不显著, 1:显著
            
            # 填充矩阵
            pairs = pairwise_results.get('pairs', {})
            for pair_key, result in pairs.items():
                # 解析模型名称
                if '_vs_' in pair_key:
                    model1, model2 = pair_key.split('_vs_')
                    
                    if model1 in models and model2 in models:
                        i = models.index(model1)
                        j = models.index(model2)
                        
                        # 存储p值
                        p_value_matrix[i, j] = result.get('p_value', np.nan)
                        p_value_matrix[j, i] = result.get('p_value', np.nan)
                        
                        # 存储显著性
                        if pairwise_results.get('adjust_pvalue', False):
                            significant = result.get('significant_after_holm', False)
                        else:
                            significant = result.get('significant', False)
                        
                        significance_matrix[i, j] = 1 if significant else 0
                        significance_matrix[j, i] = 1 if significant else 0
            
            # 创建两个子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. p值热图
            mask = np.zeros_like(p_value_matrix)
            mask[np.triu_indices_from(mask)] = True
            
            with sns.axes_style("white"):
                sns.heatmap(p_value_matrix, mask=mask, cmap='viridis_r', square=True,
                           xticklabels=models, yticklabels=models,
                           annot=True, fmt='.3f', cbar_kws={'label': 'p值'},
                           ax=ax1)
            
            ax1.set_title(f'两两比较的p值热图 ({metric})')
            ax1.set_xlabel('模型')
            ax1.set_ylabel('模型')
            
            # 2. 显著性热图
            mask = np.zeros_like(significance_matrix)
            mask[np.triu_indices_from(mask)] = True
            
            with sns.axes_style("white"):
                sns.heatmap(significance_matrix, mask=mask, cmap='coolwarm', square=True,
                           xticklabels=models, yticklabels=models,
                           annot=True, fmt='d', cbar_kws={'label': '显著性 (1=显著)'},
                           ax=ax2)
            
            ax2.set_title(f'两两比较的显著性热图 ({metric})')
            ax2.set_xlabel('模型')
            ax2.set_ylabel('模型')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f'comparison_heatmap_{metric}.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制比较热图时出错: {e}")
    
    def _plot_confidence_intervals(self, ci_results: Dict[str, Any], metric: str):
        """
        绘制置信区间图
        """
        try:
            models = []
            means = []
            lower_bounds = []
            upper_bounds = []
            
            for model, result in ci_results['models'].items():
                if result.get('valid', False):
                    models.append(model)
                    means.append(result['mean'])
                    lower_bounds.append(result['confidence_interval']['lower'])
                    upper_bounds.append(result['confidence_interval']['upper'])
            
            if not models:
                return
            
            # 计算误差条长度
            error_lower = [means[i] - lower_bounds[i] for i in range(len(models))]
            error_upper = [upper_bounds[i] - means[i] for i in range(len(models))]
            
            plt.figure(figsize=(10, 6))
            
            # 绘制误差条图
            bars = plt.bar(models, means, yerr=[error_lower, error_upper], 
                          capsize=5, alpha=0.7, color='skyblue')
            
            # 为每个条添加数值标签
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, 
                        f'{mean:.3f}', ha='center', va='bottom')
            
            # 为置信区间添加标签
            for i, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
                plt.text(i, min(lower, upper) - (max(upper_bounds) - min(lower_bounds)) * 0.05, 
                        f'[{lower:.3f}, {upper:.3f}]', ha='center', va='top', fontsize=8,
                        rotation=90)
            
            plt.title(f'各模型{metric}指标的{int(ci_results["confidence_level"]*100)}%置信区间')
            plt.xlabel('模型')
            plt.ylabel(metric)
            plt.grid(True, axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.figures_dir, f'confidence_intervals_{metric}.png'), dpi=300)
            plt.close()
            
            # 绘制森林图（更适合比较多个置信区间）
            fig, ax = plt.subplots(figsize=(8, len(models) * 0.5 + 2))
            
            # 逆序排列，使最好的模型在顶部
            if any(keyword in metric.lower() for keyword in ['mae', 'rmse', 'mape']):
                # 误差指标，值越小越好
                sorted_indices = np.argsort(means)
            else:
                # 其他指标，值越大越好
                sorted_indices = np.argsort(means)[::-1]
            
            sorted_models = [models[i] for i in sorted_indices]
            sorted_means = [means[i] for i in sorted_indices]
            sorted_lower = [lower_bounds[i] for i in sorted_indices]
            sorted_upper = [upper_bounds[i] for i in sorted_indices]
            
            y_pos = np.arange(len(sorted_models))
            
            # 绘制置信区间
            for i, (lower, upper) in enumerate(zip(sorted_lower, sorted_upper)):
                ax.plot([lower, upper], [y_pos[i], y_pos[i]], 'k-', linewidth=2)
                ax.plot(sorted_means[i], y_pos[i], 'ko', markersize=6)
            
            # 设置轴
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_models)
            ax.set_xlabel(metric)
            ax.set_title(f'各模型{metric}指标的{int(ci_results["confidence_level"]*100)}%置信区间 (森林图)')
            
            # 添加网格
            ax.grid(True, axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f'forest_plot_{metric}.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制置信区间图时出错: {e}")
    
    def _generate_html_report(self, report: Dict[str, Any]):
        """
        生成HTML格式的统计分析报告
        """
        try:
            html_template = f'''
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{report['title']}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f9f9f9;
                    }}
                    h1, h2, h3, h4 {{
                        color: #2c3e50;
                    }}
                    .header {{
                        background-color: #3498db;
                        color: white;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 30px;
                    }}
                    .section {{
                        background-color: white;
                        padding: 25px;
                        margin-bottom: 25px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .metric-section {{
                        margin-top: 40px;
                        border-top: 2px solid #3498db;
                        padding-top: 20px;
                    }}
                    .summary-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    .summary-table th, .summary-table td {{
                        padding: 12px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }}
                    .summary-table th {{
                        background-color: #f2f2f2;
                    }}
                    .highlight {{
                        background-color: #fff3cd;
                        padding: 15px;
                        border-left: 4px solid #ffc107;
                        margin: 20px 0;
                    }}
                    .significant {{
                        color: #e74c3c;
                        font-weight: bold;
                    }}
                    .insignificant {{
                        color: #27ae60;
                    }}
                    .figure-container {{
                        margin: 30px 0;
                        text-align: center;
                    }}
                    .figure-container img {{
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .figure-caption {{
                        font-style: italic;
                        margin-top: 10px;
                        color: #666;
                    }}
                    .key-findings {{
                        background-color: #d4edda;
                        padding: 20px;
                        border-radius: 8px;
                        border-left: 4px solid #28a745;
                        margin: 20px 0;
                    }}
                    .key-findings ul {{
                        margin-bottom: 0;
                    }}
                    .footer {{
                        margin-top: 50px;
                        text-align: center;
                        color: #7f8c8d;
                        font-size: 0.9em;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{report['title']}</h1>
                    <p>生成时间: {report['generated_at']}</p>
                </div>
                
                <div class="section">
                    <h2>1. 摘要</h2>
                    <table class="summary-table">
                        <tr>
                            <th>项目</th>
                            <th>描述</th>
                        </tr>
                        <tr>
                            <td>总记录数</td>
                            <td>{report['summary']['total_records']}</td>
                        </tr>
                        <tr>
                            <td>模型数量</td>
                            <td>{len(report['summary']['models'])}</td>
                        </tr>
                        <tr>
                            <td>评估指标</td>
                            <td>{', '.join(report['summary']['metrics'])}</td>
                        </tr>
                        <tr>
                            <td>各模型运行次数</td>
                            <td>
                                <ul>
                                    {''.join(f'<li>{model}: {runs}次</li>' for model, runs in report['summary']['runs_per_model'].items())}
                                </ul>
                            </td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>2. 关键发现</h2>
                    <div class="key-findings">
                        <ul>
                            {''.join(f'<li>{finding}</li>' for finding in report['key_findings'])}
                        </ul>
                    </div>
                </div>
            '''
            
            # 添加每个指标的详细分析
            for metric, analysis in report['metrics_analysis'].items():
                html_template += f'''
                <div class="section metric-section">
                    <h2>3. 指标分析: {metric}</h2>
                    
                    <h3>3.1 基本统计信息</h3>
                    <table class="summary-table">
                        <tr>
                            <th>模型</th>
                            <th>均值</th>
                            <th>标准差</th>
                            <th>中位数</th>
                            <th>最小值</th>
                            <th>最大值</th>
                            <th>样本数</th>
                        </tr>
                '''
                
                # 基本统计表格
                for model, stats in analysis['basic_statistics'].items():
                    html_template += f'''
                        <tr>
                            <td>{model}</td>
                            <td>{stats['mean']:.4f}</td>
                            <td>{stats['std']:.4f}</td>
                            <td>{stats['median']:.4f}</td>
                            <td>{stats['min']:.4f}</td>
                            <td>{stats['max']:.4f}</td>
                            <td>{stats['count']}</td>
                        </tr>
                    '''
                
                html_template += '''
                    </table>
                    
                    <h3>3.2 正态性检验</h3>
                '''
                
                # 正态性检验结果
                for model, result in analysis['normality_tests'].items():
                    if result.get('valid', False):
                        is_normal_str = '是' if result['is_normal'] else '否'
                        html_template += f'''
                        <p><strong>{model}:</strong> {result['test_name']}检验 - 统计量={result['statistic']:.4f}, 
                        p值={result['p_value']:.4f}, 服从正态分布: <span class="{'significant' if result['is_normal'] else 'insignificant'}">{is_normal_str}</span></p>
                        '''
                
                html_template += '''
                    <h3>3.3 方差分析 (ANOVA)</h3>
                '''
                
                # ANOVA结果
                anova = analysis['anova_result']
                if anova:
                    significant_str = '是' if anova['significant'] else '否'
                    html_template += f'''
                    <p>F统计量: {anova['f_statistic']:.4f}, p值: {anova['p_value']:.4f}, 
                    差异显著: <span class="{'significant' if anova['significant'] else 'insignificant'}">{significant_str}</span></p>
                    <p>效应量 (eta-squared): {anova['eta_squared']:.4f} ({anova['effect_size_interpretation']})</p>
                    '''
                    
                    # Tukey HSD结果
                    if anova.get('post_hoc'):
                        html_template += '''
                        <h4>3.3.1 Tukey HSD事后检验结果</h4>
                        <div class="highlight">
                            <p>显著差异的模型对:</p>
                            <ul>
                        '''
                        for pair in anova['post_hoc'].get('significant_pairs', []):
                            html_template += f'''
                                <li>{pair['group1']} vs {pair['group2']}: 差异={pair['meandiff']:.4f}, p值={pair['p-adj']:.4f}</li>
                            '''
                        html_template += '''
                            </ul>
                        </div>
                        '''
                
                html_template += '''
                    <h3>3.4 两两比较结果</h3>
                '''
                
                # 两两比较结果
                pairwise = analysis['pairwise_comparisons']
                if pairwise:
                    significant_count = pairwise.get('summary', {}).get('significant_pairs', 0)
                    html_template += f'''
                    <p>共分析 {pairwise.get('summary', {}).get('total_pairs', 0)} 对模型，
                    发现 <span class="significant">{significant_count}</span> 对存在统计学显著差异（经过多重比较校正）。</p>
                    '''
                
                # 添加图片链接
                html_template += f'''
                    <h3>3.5 可视化结果</h3>
                    
                    <div class="figure-container">
                        <h4>箱线图</h4>
                        <img src="../figures/boxplot_{metric}.png" alt="箱线图">
                        <p class="figure-caption">不同模型的{metric}指标分布箱线图</p>
                    </div>
                    
                    <div class="figure-container">
                        <h4>置信区间图</h4>
                        <img src="../figures/confidence_intervals_{metric}.png" alt="置信区间图">
                        <p class="figure-caption">各模型{metric}指标的95%置信区间</p>
                    </div>
                    
                    <div class="figure-container">
                        <h4>森林图</h4>
                        <img src="../figures/forest_plot_{metric}.png" alt="森林图">
                        <p class="figure-caption">各模型{metric}指标的95%置信区间（森林图）</p>
                    </div>
                    
                    <div class="figure-container">
                        <h4>两两比较热图</h4>
                        <img src="../figures/comparison_heatmap_{metric}.png" alt="比较热图">
                        <p class="figure-caption">模型两两比较的p值和显著性热图</p>
                    </div>
                </div>
                '''
            
            # 添加页脚
            html_template += '''
                <div class="footer">
                    <p>本报告由统计显著性分析工具自动生成</p>
                </div>
            </body>
            </html>
            '''
            
            # 保存HTML报告
            html_file = os.path.join(self.output_dir, 'statistical_analysis_report.html')
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_template)
                
            logger.info(f"HTML格式报告已生成: {html_file}")
            
        except Exception as e:
            logger.error(f"生成HTML报告时出错: {e}")

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='执行统计显著性分析')
    parser.add_argument('--results_dir', type=str, default='../results/evaluations',
                        help='评估结果文件目录')
    parser.add_argument('--output', type=str, default='../results/statistical_analysis',
                        help='分析结果输出目录')
    parser.add_argument('--metric', type=str, help='要分析的特定指标')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='显著性水平 (默认: 0.05)')
    parser.add_argument('--no_plots', action='store_true',
                        help='不生成可视化图表')
    parser.add_argument('--detailed_report', action='store_true',
                        help='生成详细报告')
    
    args = parser.parse_args()
    
    # 创建统计分析器
    analyzer = StatisticalAnalyzer(args.output)
    
    # 加载结果
    if not analyzer.load_results(args.results_dir):
        logger.error("加载结果失败，程序退出")
        sys.exit(1)
    
    logger.info("开始执行统计显著性分析...")
    
    # 执行详细报告生成或特定分析
    if args.detailed_report:
        analyzer.generate_detailed_report()
    else:
        # 如果指定了特定指标
        if args.metric:
            if args.metric not in analyzer.metrics:
                logger.error(f"指定的指标 {args.metric} 不存在")
                logger.info(f"可用指标: {', '.join(analyzer.metrics)}")
                sys.exit(1)
            
            # 执行基本分析
            logger.info(f"分析指标: {args.metric}")
            
            # 正态性检验
            analyzer.check_normality(args.metric, args.alpha)
            
            # 方差齐性检验
            analyzer.check_homogeneity(args.metric, args.alpha)
            
            # 计算置信区间
            analyzer.compute_confidence_intervals(args.metric)
            
            # ANOVA检验
            analyzer.anova_test(args.metric, args.alpha)
            
            # 两两比较
            analyzer.run_all_paired_tests(args.metric, args.alpha)
        else:
            # 分析所有指标
            for metric in analyzer.metrics:
                logger.info(f"分析指标: {metric}")
                
                # 正态性检验
                analyzer.check_normality(metric, args.alpha)
                
                # 方差齐性检验
                analyzer.check_homogeneity(metric, args.alpha)
                
                # 计算置信区间
                analyzer.compute_confidence_intervals(metric)
                
                # ANOVA检验
                analyzer.anova_test(metric, args.alpha)
                
                # 两两比较
                analyzer.run_all_paired_tests(metric, args.alpha)
    
    logger.info("统计显著性分析完成！")
    logger.info(f"分析结果保存在: {args.output}")

if __name__ == "__main__":
    main()