#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多次运行实验脚本

此脚本用于执行模型的多次运行实验，收集统计结果，计算平均值和标准差，
并生成可视化报告，确保实验结果的统计显著性和可复现性。
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import ExperimentConfig
from evaluation.metrics import compute_confidence_intervals
from evaluation.statistical_significance import StatisticalSignificanceAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multiple_runs_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultipleRunsExperiment:
    """
    多次运行实验管理器
    """
    def __init__(self, config_file: str, models: List[str], num_runs: int = 5):
        """
        初始化多次运行实验
        
        Args:
            config_file: 配置文件路径
            models: 要运行的模型列表
            num_runs: 每个模型运行的次数
        """
        self.config_file = config_file
        self.models = models
        self.num_runs = num_runs
        
        # 创建实验目录
        self.experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            Path(config_file).parent.parent, "experiments", self.experiment_id
        )
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 创建结果目录
        self.results_dir = os.path.join(self.experiment_dir, "results")
        self.figures_dir = os.path.join(self.experiment_dir, "figures")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # 保存实验配置
        self.experiment_config = {
            "experiment_id": self.experiment_id,
            "config_file": config_file,
            "models": models,
            "num_runs": num_runs,
            "start_time": datetime.datetime.now().isoformat()
        }
        
        # 保存配置文件
        with open(os.path.join(self.experiment_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.experiment_config, f, indent=2, ensure_ascii=False)
        
        # 记录日志
        log_file = os.path.join(self.experiment_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"创建多次运行实验: {self.experiment_id}")
        logger.info(f"实验目录: {self.experiment_dir}")
        logger.info(f"要运行的模型: {', '.join(models)}")
        logger.info(f"每个模型运行次数: {num_runs}")
        
        # 存储所有运行结果
        self.all_results = {model: [] for model in models}
        self.aggregated_results = {}
    
    def run_experiments(self):
        """
        运行所有模型的多次实验
        """
        for model in self.models:
            logger.info(f"\n========== 开始运行模型: {model} ==========")
            
            for run_id in range(self.num_runs):
                logger.info(f"\n开始运行 {model} (运行 {run_id + 1}/{self.num_runs})")
                
                # 设置随机种子
                seed = run_id  # 每个运行使用不同的种子以增加统计显著性
                
                # 构建运行命令
                command = [
                    "python",
                    os.path.join("code", "training", "baseline_model_trainer.py"),
                    "--config", self.config_file,
                    "--model", model,
                    "--runs", "1",
                    "--gpu", "0"  # 可以根据需要调整GPU索引
                ]
                
                # 构建命令字符串
                command_str = " ".join(command)
                logger.info(f"执行命令: {command_str}")
                
                # 设置环境变量以确保可复现性
                env = os.environ.copy()
                env["PYTHONHASHSEED"] = str(seed)
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 确保cuDNN确定性
                
                try:
                    # 运行命令
                    start_time = time.time()
                    result = self._run_command(command_str, env)
                    end_time = time.time()
                    
                    # 记录运行时间
                    run_time = end_time - start_time
                    logger.info(f"运行 {run_id + 1} 完成，耗时: {run_time:.2f} 秒")
                    
                    # 解析结果
                    run_result = self._parse_run_result(model, run_id, result)
                    
                    # 添加运行时间
                    run_result["run_time"] = run_time
                    
                    # 保存单次运行结果
                    self._save_run_result(model, run_id, run_result)
                    
                    # 添加到所有结果
                    self.all_results[model].append(run_result)
                    
                except Exception as e:
                    logger.error(f"运行 {run_id + 1} 失败: {str(e)}")
                    # 保存失败结果
                    error_result = {
                        "model": model,
                        "run_id": run_id,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    self.all_results[model].append(error_result)
                    self._save_run_result(model, run_id, error_result)
            
            logger.info(f"模型 {model} 的所有运行完成")
    
    def _run_command(self, command: str, env: Dict[str, str] = None) -> str:
        """
        运行命令并返回输出
        
        Args:
            command: 要运行的命令
            env: 环境变量字典
            
        Returns:
            str: 命令输出
        """
        import subprocess
        
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            env=env
        )
        
        # 合并标准输出和错误输出
        output = result.stdout + "\n" + result.stderr
        
        # 检查返回码
        if result.returncode != 0:
            raise Exception(f"命令执行失败，返回码: {result.returncode}\n输出: {output}")
        
        return output
    
    def _parse_run_result(self, model: str, run_id: int, output: str) -> Dict[str, Any]:
        """
        解析运行输出，提取结果
        
        Args:
            model: 模型名称
            run_id: 运行ID
            output: 命令输出
            
        Returns:
            Dict: 解析后的运行结果
        """
        # 在实际应用中，这里需要根据baseline_model_trainer.py的输出格式进行解析
        # 这里提供一个简化版本
        
        # 查找测试结果文件
        config = ExperimentConfig(self.config_file)
        output_base_dir = config.output.save_dir
        model_dir = os.path.join(output_base_dir, model)
        
        # 查找最新的模型输出目录
        if os.path.exists(model_dir):
            subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) 
                      if os.path.isdir(os.path.join(model_dir, d))]
            if subdirs:
                latest_dir = max(subdirs, key=os.path.getmtime)
                test_results_file = os.path.join(latest_dir, "test_results.json")
                
                if os.path.exists(test_results_file):
                    try:
                        with open(test_results_file, "r", encoding="utf-8") as f:
                            test_results = json.load(f)
                        
                        # 返回解析后的结果
                        return {
                            "model": model,
                            "run_id": run_id,
                            "status": "success",
                            "test_metrics": test_results.get("test_metrics", {}),
                            "best_epoch": test_results.get("best_epoch", 0),
                            "output_dir": latest_dir,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    except Exception as e:
                        logger.error(f"解析测试结果文件失败: {str(e)}")
        
        # 如果找不到结果文件，尝试从输出中解析
        # 这是一个后备方案
        result = {
            "model": model,
            "run_id": run_id,
            "status": "success",
            "test_metrics": {},
            "best_epoch": 0,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # 这里可以添加从输出文本中提取指标的代码
        # 例如使用正则表达式
        
        return result
    
    def _save_run_result(self, model: str, run_id: int, result: Dict[str, Any]):
        """
        保存单次运行结果
        
        Args:
            model: 模型名称
            run_id: 运行ID
            result: 运行结果
        """
        # 创建模型结果目录
        model_results_dir = os.path.join(self.results_dir, model)
        os.makedirs(model_results_dir, exist_ok=True)
        
        # 保存结果
        result_file = os.path.join(model_results_dir, f"run_{run_id}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"运行结果已保存到: {result_file}")
    
    def aggregate_results(self):
        """
        聚合所有运行结果，计算统计指标
        """
        logger.info("\n开始聚合结果...")
        
        for model in self.models:
            logger.info(f"\n聚合模型 {model} 的结果...")
            
            # 获取成功运行的结果
            successful_runs = [run for run in self.all_results[model] 
                              if run.get("status") == "success"]
            
            if not successful_runs:
                logger.warning(f"模型 {model} 没有成功运行的结果")
                continue
            
            logger.info(f"成功运行次数: {len(successful_runs)}/{self.num_runs}")
            
            # 获取所有指标名称
            metrics = set()
            for run in successful_runs:
                metrics.update(run.get("test_metrics", {}).keys())
            
            # 初始化聚合结果
            aggregated = {
                "model": model,
                "total_runs": self.num_runs,
                "successful_runs": len(successful_runs),
                "metrics": {}
            }
            
            # 聚合每个指标
            for metric in metrics:
                # 收集所有运行的指标值
                values = [run["test_metrics"].get(metric, np.nan) 
                         for run in successful_runs]
                
                # 过滤NaN值
                values = [v for v in values if not np.isnan(v)]
                
                if values:
                    aggregated["metrics"][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                        "all_values": values
                    }
                    
                    logger.info(f"  {metric.upper()}: {np.mean(values):.6f} ± {np.std(values):.6f} "
                              f"(min={np.min(values):.6f}, max={np.max(values):.6f})")
            
            # 聚合运行时间
            if all("run_time" in run for run in successful_runs):
                run_times = [run["run_time"] for run in successful_runs]
                aggregated["run_time"] = {
                    "mean": np.mean(run_times),
                    "std": np.std(run_times)
                }
                logger.info(f"  平均运行时间: {np.mean(run_times):.2f} ± {np.std(run_times):.2f} 秒")
            
            # 保存聚合结果
            self.aggregated_results[model] = aggregated
        
        # 保存所有聚合结果
        with open(os.path.join(self.experiment_dir, "aggregated_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.aggregated_results, f, indent=2, ensure_ascii=False)
        
        logger.info("\n聚合结果已保存到: aggregated_results.json")
    
    def perform_statistical_analysis(self):
        """
        执行统计显著性分析
        """
        logger.info("\n开始统计显著性分析...")
        
        # 获取所有模型的聚合结果
        models = list(self.aggregated_results.keys())
        if len(models) < 2:
            logger.warning("模型数量少于2，无法进行统计显著性分析")
            return
        
        # 获取所有共同的指标
        common_metrics = set.intersection(
            *[set(agg["metrics"].keys()) for agg in self.aggregated_results.values()]
        )
        
        # 创建统计分析器
        analyzer = StatisticalSignificanceAnalyzer()
        
        # 为每个指标进行分析
        statistical_results = {}
        
        for metric in common_metrics:
            logger.info(f"\n分析指标: {metric}")
            
            # 收集每个模型的指标值
            model_values = {}
            for model in models:
                if metric in self.aggregated_results[model]["metrics"]:
                    model_values[model] = self.aggregated_results[model]["metrics"][metric]["all_values"]
            
            # 执行方差分析（如果有多个模型）
            if len(model_values) >= 3:
                anova_result = analyzer.perform_anova(
                    list(model_values.values()), 
                    list(model_values.keys())
                )
                statistical_results[f"{metric}_anova"] = anova_result
                
                logger.info(f"  ANOVA 结果: F统计量={anova_result['f_statistic']:.4f}, "
                          f"p值={anova_result['p_value']:.6f}")
                
                if anova_result["p_value"] < 0.05:
                    logger.info(f"  发现显著差异 (p < 0.05)，执行事后检验...")
                    
                    # 执行Tukey HSD事后检验
                    tukey_result = analyzer.perform_tukey_hsd(
                        list(model_values.values()), 
                        list(model_values.keys())
                    )
                    statistical_results[f"{metric}_tukey"] = tukey_result
                    
                    # 打印显著差异
                    significant_pairs = []
                    for comparison in tukey_result["comparisons"]:
                        if comparison["p_value"] < 0.05:
                            significant_pairs.append(comparison)
                            logger.info(f"    {comparison['group1']} vs {comparison['group2']}: "
                                      f"p值={comparison['p_value']:.6f}, "
                                      f"差异={comparison['mean_diff']:.6f}")
            
            # 执行所有模型对之间的t检验
            t_test_results = []
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i < j and model1 in model_values and model2 in model_values:
                        # 配对t检验（假设数据是配对的）
                        t_test = analyzer.perform_t_test(
                            model_values[model1], 
                            model_values[model2],
                            paired=True
                        )
                        
                        t_test_results.append({
                            "model1": model1,
                            "model2": model2,
                            "t_statistic": t_test["t_statistic"],
                            "p_value": t_test["p_value"],
                            "significant": t_test["p_value"] < 0.05
                        })
                        
                        logger.info(f"  t检验 ({model1} vs {model2}): "
                                  f"t={t_test['t_statistic']:.4f}, "
                                  f"p={t_test['p_value']:.6f}, "
                                  f"显著={'是' if t_test['p_value'] < 0.05 else '否'}")
            
            statistical_results[f"{metric}_t_tests"] = t_test_results
        
        # 保存统计分析结果
        with open(os.path.join(self.experiment_dir, "statistical_analysis_results.json"), "w", encoding="utf-8") as f:
            json.dump(statistical_results, f, indent=2, ensure_ascii=False)
        
        logger.info("\n统计分析结果已保存到: statistical_analysis_results.json")
    
    def generate_visualizations(self):
        """
        生成可视化图表
        """
        logger.info("\n开始生成可视化图表...")
        
        # 检查是否有聚合结果
        if not self.aggregated_results:
            logger.warning("没有聚合结果，无法生成可视化图表")
            return
        
        # 获取所有模型和指标
        models = list(self.aggregated_results.keys())
        common_metrics = set.intersection(
            *[set(agg["metrics"].keys()) for agg in self.aggregated_results.values()]
        )
        
        # 1. 箱线图 - 比较各模型的指标分布
        self._plot_boxplots(models, common_metrics)
        
        # 2. 条形图 - 比较各模型的平均性能
        self._plot_barplots(models, common_metrics)
        
        # 3. 雷达图 - 综合比较各模型的所有指标
        self._plot_radar_chart(models, common_metrics)
        
        # 4. 折线图 - 比较各模型在不同时间步的性能（如果有时间序列数据）
        # 这里假设我们有时间序列数据，实际使用时需要根据具体情况修改
        
        logger.info("\n可视化图表已生成并保存到figures目录")
    
    def _plot_boxplots(self, models: List[str], metrics: List[str]):
        """
        绘制箱线图
        
        Args:
            models: 模型列表
            metrics: 指标列表
        """
        for metric in metrics:
            # 准备数据
            data = []
            labels = []
            
            for model in models:
                if metric in self.aggregated_results[model]["metrics"]:
                    values = self.aggregated_results[model]["metrics"][metric]["all_values"]
                    data.append(values)
                    labels.append(model)
            
            if len(data) >= 2:
                plt.figure(figsize=(10, 6))
                box = plt.boxplot(data, labels=labels, patch_artist=True)
                
                # 设置颜色
                colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                
                plt.title(f'{metric.upper()} 指标分布')
                plt.xlabel('模型')
                plt.ylabel(metric.upper())
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # 保存图表
                plt.tight_layout()
                plt.savefig(os.path.join(self.figures_dir, f"boxplot_{metric}.png"), dpi=300)
                plt.close()
                
                logger.info(f"生成箱线图: {metric}")
    
    def _plot_barplots(self, models: List[str], metrics: List[str]):
        """
        绘制条形图
        
        Args:
            models: 模型列表
            metrics: 指标列表
        """
        for metric in metrics:
            # 准备数据
            means = []
            stds = []
            labels = []
            
            for model in models:
                if metric in self.aggregated_results[model]["metrics"]:
                    means.append(self.aggregated_results[model]["metrics"][metric]["mean"])
                    stds.append(self.aggregated_results[model]["metrics"][metric]["std"])
                    labels.append(model)
            
            if means:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(labels, means, yerr=stds, capsize=5)
                
                # 设置颜色
                colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                plt.title(f'{metric.upper()} 平均性能比较')
                plt.xlabel('模型')
                plt.ylabel(metric.upper())
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                
                # 在条形上添加数值标签
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{means[i]:.4f}\n±{stds[i]:.4f}',
                             ha='center', va='bottom', rotation=0)
                
                # 保存图表
                plt.tight_layout()
                plt.savefig(os.path.join(self.figures_dir, f"barplot_{metric}.png"), dpi=300)
                plt.close()
                
                logger.info(f"生成条形图: {metric}")
    
    def _plot_radar_chart(self, models: List[str], metrics: List[str]):
        """
        绘制雷达图
        
        Args:
            models: 模型列表
            metrics: 指标列表
        """
        # 对于雷达图，我们需要对指标进行归一化
        # 首先确定哪些指标是越小越好，哪些是越大越好
       越小越好 = ['mae', 'rmse', 'mse', 'mape', 'mre', 'smape', 'wape']
       越大越好 = ['r2']
        
        # 选择适合雷达图的指标
        radar_metrics = [m for m in metrics if m in 越小越好 or m in 越大越好]
        
        if len(radar_metrics) < 3:
            logger.warning("指标数量不足，无法生成雷达图")
            return
        
        # 准备数据
        model_data = {}
        for model in models:
            model_data[model] = []
            for metric in radar_metrics:
                if metric in self.aggregated_results[model]["metrics"]:
                    model_data[model].append(self.aggregated_results[model]["metrics"][metric]["mean"])
        
        # 计算每个指标的最大值和最小值用于归一化
        metric_max = {}
        metric_min = {}
        for i, metric in enumerate(radar_metrics):
            values = [model_data[model][i] for model in models if i < len(model_data[model])]
            if values:
                metric_max[metric] = max(values)
                metric_min[metric] = min(values)
        
        # 归一化数据
        normalized_data = {}
        for model in models:
            normalized_data[model] = []
            for i, metric in enumerate(radar_metrics):
                if i < len(model_data[model]) and metric in metric_max:
                    value = model_data[model][i]
                    if metric in 越小越好:
                        # 越小越好的指标，归一化为1-0
                        if metric_max[metric] > metric_min[metric]:
                            norm = 1 - (value - metric_min[metric]) / (metric_max[metric] - metric_min[metric])
                        else:
                            norm = 1.0
                    else:  # 越大越好的指标，归一化为0-1
                        if metric_max[metric] > metric_min[metric]:
                            norm = (value - metric_min[metric]) / (metric_max[metric] - metric_min[metric])
                        else:
                            norm = 0.0
                    normalized_data[model].append(norm)
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # 为每个模型绘制雷达图
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for i, model in enumerate(models):
            if model in normalized_data and len(normalized_data[model]) == len(radar_metrics):
                values = normalized_data[model] + normalized_data[model][:1]  # 闭合雷达图
                ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=model)
                ax.fill(angles, values, color=colors[i], alpha=0.25)
        
        # 设置雷达图的标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in radar_metrics])
        
        # 添加图例
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('模型性能雷达图（归一化）', size=15, y=1.1)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(self.figures_dir, "radar_chart.png"), dpi=300)
        plt.close()
        
        logger.info("生成雷达图")
    
    def generate_summary_report(self):
        """
        生成实验总结报告
        """
        logger.info("\n生成实验总结报告...")
        
        # 创建报告
        report = {
            "experiment_id": self.experiment_id,
            "experiment_config": self.experiment_config,
            "aggregated_results": self.aggregated_results,
            "conclusion": self._generate_conclusion(),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # 保存报告
        with open(os.path.join(self.experiment_dir, "summary_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        self._generate_html_report(report)
        
        logger.info("实验总结报告已生成")
    
    def _generate_conclusion(self) -> Dict[str, Any]:
        """
        根据实验结果生成结论
        
        Returns:
            Dict: 结论字典
        """
        conclusion = {
            "best_performing_models": {},
            "key_findings": [],
            "recommendations": []
        }
        
        # 获取所有共同的指标
        common_metrics = set.intersection(
            *[set(agg["metrics"].keys()) for agg in self.aggregated_results.values()]
        )
        
        # 确定哪些指标是越小越好，哪些是越大越好
        越小越好 = ['mae', 'rmse', 'mse', 'mape', 'mre', 'smape', 'wape']
        越大越好 = ['r2']
        
        # 确定每个指标的最佳模型
        for metric in common_metrics:
            best_model = None
            best_value = None
            
            for model, agg in self.aggregated_results.items():
                if metric in agg["metrics"]:
                    current_value = agg["metrics"][metric]["mean"]
                    
                    if best_value is None:
                        best_model = model
                        best_value = current_value
                    elif metric in 越小越好:
                        if current_value < best_value:
                            best_model = model
                            best_value = current_value
                    elif metric in 越大越好:
                        if current_value > best_value:
                            best_model = model
                            best_value = current_value
            
            if best_model:
                conclusion["best_performing_models"][metric] = {
                    "model": best_model,
                    "value": best_value
                }
        
        # 生成关键发现
        if conclusion["best_performing_models"]:
            # 找出在最多指标上表现最好的模型
            model_counts = {}
            for metric, info in conclusion["best_performing_models"].items():
                model = info["model"]
                model_counts[model] = model_counts.get(model, 0) + 1
            
            if model_counts:
                overall_best_model = max(model_counts.items(), key=lambda x: x[1])[0]
                conclusion["key_findings"].append(
                    f"总体表现最佳的模型是 {overall_best_model}，在 {model_counts[overall_best_model]} 个指标上表现最优。"
                )
        
        # 检查是否有显著的性能差异
        # 这部分需要根据统计分析结果来生成
        # 这里简化处理
        
        # 生成建议
        conclusion["recommendations"].append(
            "基于实验结果，建议在实际应用中使用表现最佳的模型。"
        )
        
        conclusion["recommendations"].append(
            "对于超参数优化，可以进一步探索性能表现较好的模型的参数空间。"
        )
        
        conclusion["recommendations"].append(
            "建议在更大规模的数据集上进行验证，以确保模型的泛化能力。"
        )
        
        return conclusion
    
    def _generate_html_report(self, report: Dict[str, Any]):
        """
        生成HTML格式的报告
        
        Args:
            report: 报告数据
        """
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>实验总结报告 - {report['experiment_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .metric-best {{ font-weight: bold; color: #e74c3c; }}
                .figure-container {{ margin: 20px 0; text-align: center; }}
                .figure-container img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>实验总结报告</h1>
                
                <div class="summary-box">
                    <h2>实验信息</h2>
                    <p><strong>实验ID:</strong> {report['experiment_id']}</p>
                    <p><strong>开始时间:</strong> {report['experiment_config']['start_time']}</p>
                    <p><strong>结束时间:</strong> {report['timestamp']}</p>
                    <p><strong>运行模型:</strong> {', '.join(report['experiment_config']['models'])}</p>
                    <p><strong>每个模型运行次数:</strong> {report['experiment_config']['num_runs']}</p>
                </div>
        """
        
        # 添加聚合结果表格
        html_content += """
                <h2>聚合结果</h2>
        """
        
        # 获取所有共同的指标
        models = list(report['aggregated_results'].keys())
        if models:
            common_metrics = set.intersection(
                *[set(agg["metrics"].keys()) for agg in report['aggregated_results'].values()]
            )
            
            # 对于每个指标创建一个表格
            for metric in common_metrics:
                html_content += f"""
                <h3>{metric.upper()} 指标比较</h3>
                <table>
                    <tr>
                        <th>模型</th>
                        <th>平均值</th>
                        <th>标准差</th>
                        <th>最小值</th>
                        <th>最大值</th>
                        <th>中位数</th>
                        <th>成功运行次数</th>
                    </tr>
                """
                
                # 找出当前指标的最佳值
                best_value = None
                best_model = None
                for model in models:
                    if metric in report['aggregated_results'][model]['metrics']:
                        current_value = report['aggregated_results'][model]['metrics'][metric]['mean']
                        if best_value is None or current_value < best_value:  # 假设指标越小越好
                            best_value = current_value
                            best_model = model
                
                # 添加每个模型的数据
                for model in models:
                    if metric in report['aggregated_results'][model]['metrics']:
                        metrics_data = report['aggregated_results'][model]['metrics'][metric]
                        is_best = (model == best_model)
                        html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td class="{'metric-best' if is_best else ''}">{metrics_data['mean']:.6f}</td>
                        <td>{metrics_data['std']:.6f}</td>
                        <td>{metrics_data['min']:.6f}</td>
                        <td>{metrics_data['max']:.6f}</td>
                        <td>{metrics_data['median']:.6f}</td>
                        <td>{report['aggregated_results'][model]['successful_runs']}</td>
                    </tr>
                        """
                
                html_content += "</table>\n"
        
        # 添加结论部分
        html_content += """
                <h2>结论</h2>
        """
        
        # 最佳表现模型
        if 'best_performing_models' in report['conclusion']:
            html_content += """
                <h3>最佳表现模型</h3>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>最佳模型</th>
                        <th>最佳值</th>
                    </tr>
            """
            
            for metric, info in report['conclusion']['best_performing_models'].items():
                html_content += f"""
                    <tr>
                        <td>{metric.upper()}</td>
                        <td>{info['model']}</td>
                        <td>{info['value']:.6f}</td>
                    </tr>
                """
            
            html_content += "</table>\n"
        
        # 关键发现
        if 'key_findings' in report['conclusion']:
            html_content += """
                <h3>关键发现</h3>
                <ul>
            """
            
            for finding in report['conclusion']['key_findings']:
                html_content += f"<li>{finding}</li>\n"
            
            html_content += "</ul>\n"
        
        # 建议
        if 'recommendations' in report['conclusion']:
            html_content += """
                <h3>建议</h3>
                <ul>
            """
            
            for recommendation in report['conclusion']['recommendations']:
                html_content += f"<li>{recommendation}</li>\n"
            
            html_content += "</ul>\n"
        
        # 添加图表
        html_content += """
                <h2>可视化图表</h2>
                <div class="figure-container">
        """
        
        # 查找并添加图表
        figures_dir = self.figures_dir
        if os.path.exists(figures_dir):
            for filename in os.listdir(figures_dir):
                if filename.endswith('.png'):
                    figure_path = os.path.join(figures_dir, filename)
                    relative_path = os.path.relpath(figure_path, self.experiment_dir)
                    figure_name = filename.replace('_', ' ').replace('.png', '').title()
                    
                    html_content += f"""
                    <div>
                        <h4>{figure_name}</h4>
                        <img src="{relative_path}" alt="{figure_name}">
                    </div>
                    """
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML报告
        with open(os.path.join(self.experiment_dir, "summary_report.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
    
    def run(self):
        """
        运行完整的实验流程
        """
        try:
            logger.info("开始执行多次运行实验...")
            
            # 运行实验
            self.run_experiments()
            
            # 聚合结果
            self.aggregate_results()
            
            # 执行统计分析
            self.perform_statistical_analysis()
            
            # 生成可视化
            self.generate_visualizations()
            
            # 生成报告
            self.generate_summary_report()
            
            logger.info("多次运行实验完成！")
            logger.info(f"实验总结报告: {os.path.join(self.experiment_dir, 'summary_report.html')}")
            
        except Exception as e:
            logger.error(f"实验执行失败: {str(e)}")
            raise


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="多次运行实验脚本")
    parser.add_argument("--config", type=str, default="../configs/experiment_config.yaml",
                        help="实验配置文件路径")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["GCN", "LSTM", "STGCN", "T-GCN", "GCNLSTMHybrid"],
                        help="要运行的模型列表")
    parser.add_argument("--runs", type=int, default=5,
                        help="每个模型运行的次数")
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_arguments()
    
    # 创建并运行多次实验
    experiment = MultipleRunsExperiment(
        config_file=args.config,
        models=args.models,
        num_runs=args.runs
    )
    
    experiment.run()


if __name__ == "__main__":
    main()