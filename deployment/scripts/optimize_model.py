#!/usr/bin/env python3
"""
模型压缩和部署优化主脚本
"""

import argparse
import os
import sys
import json
import yaml
import time
import torch
from pathlib import Path

# 添加部署模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compression.model_pruning import StructuredPruner, UnstructuredPruner, PruningScheduler
from compression.knowledge_distillation import KnowledgeDistiller, ProgressiveDistiller
from quantization.model_quantization import QuantizationOptimizer
from caching.model_cache import ModelCache, ModelPreloader
from optimization.deployment_architecture import DeploymentOrchestrator, SystemAnalyzer
from optimization.memory_optimization import MemoryOptimizer, ResourceMonitor


class ModelOptimizationPipeline:
    """模型优化流水线"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_logging()
        
    def load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def setup_logging(self):
        """设置日志"""
        import logging
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_pruning(self, model, train_loader=None, test_data=None, test_labels=None):
        """运行模型剪枝"""
        self.logger.info("开始模型剪枝优化")
        
        pruning_config = self.config.get('pruning', {})
        pruning_type = pruning_config.get('type', 'unstructured')
        
        if pruning_type == 'structured':
            pruner = StructuredPruner(model, pruning_ratio=pruning_config.get('ratio', 0.3))
            pruned_model = pruner.prune_model(test_data, test_labels)
        elif pruning_type == 'unstructured':
            pruner = UnstructuredPruner(model, sparsity_ratio=pruning_config.get('ratio', 0.5))
            pruned_model = pruner.prune_model()
            if train_loader:
                pruner.fine_tune(train_loader, epochs=pruning_config.get('fine_tune_epochs', 10))
        elif pruning_type == 'progressive':
            scheduler = PruningScheduler(
                model,
                initial_sparsity=pruning_config.get('initial_sparsity', 0.1),
                final_sparsity=pruning_config.get('final_sparsity', 0.7),
                steps=pruning_config.get('steps', 10)
            )
            pruned_model = scheduler.progressive_prune(train_loader, test_data, test_labels)
            
            # 保存剪枝历史图
            scheduler.plot_pruning_history("pruning_history.png")
        else:
            self.logger.error(f"不支持的剪枝类型: {pruning_type}")
            return model
        
        # 保存剪枝结果
        save_dir = self.config.get('output', {}).get('pruning_dir', './output/pruning')
        os.makedirs(save_dir, exist_ok=True)
        
        from compression.model_pruning import save_pruning_results
        save_pruning_results(pruner, pruned_model, save_dir)
        
        self.logger.info("模型剪枝完成")
        return pruned_model
    
    def run_distillation(self, teacher_model, student_model, train_loader, val_loader=None):
        """运行知识蒸馏"""
        self.logger.info("开始知识蒸馏")
        
        distillation_config = self.config.get('distillation', {})
        distillation_type = distillation_config.get('type', 'standard')
        
        if distillation_type == 'standard':
            distiller = KnowledgeDistiller(
                teacher_model, student_model,
                temperature=distillation_config.get('temperature', 4.0),
                alpha=distillation_config.get('alpha', 0.7),
                beta=distillation_config.get('beta', 0.3)
            )
            distilled_model = distiller.distill(
                train_loader, val_loader,
                epochs=distillation_config.get('epochs', 50),
                lr=distillation_config.get('learning_rate', 0.001)
            )
            
        elif distillation_type == 'progressive':
            teacher_models = distillation_config.get('teacher_models', [teacher_model])
            progressive_distiller = ProgressiveDistiller(
                teacher_models, student_model,
                temperatures=distillation_config.get('temperatures', [1.0, 2.0, 4.0, 8.0]),
                alphas=distillation_config.get('alphas', [0.9, 0.7, 0.5, 0.3])
            )
            distilled_model = progressive_distiller.progressive_distill(
                train_loader, val_loader,
                epochs_per_stage=distillation_config.get('epochs_per_stage', 25)
            )
        else:
            self.logger.error(f"不支持的蒸馏类型: {distillation_type}")
            return student_model
        
        # 保存蒸馏结果
        save_dir = self.config.get('output', {}).get('distillation_dir', './output/distillation')
        os.makedirs(save_dir, exist_ok=True)
        
        from compression.knowledge_distillation import save_distillation_results
        save_distillation_results(distiller, distilled_model, save_dir)
        
        self.logger.info("知识蒸馏完成")
        return distilled_model
    
    def run_quantization(self, model, train_loader, val_loader, test_data, test_labels):
        """运行模型量化"""
        self.logger.info("开始模型量化")
        
        quantization_config = self.config.get('quantization', {})
        optimizer = QuantizationOptimizer(model)
        
        # 运行量化优化
        results = optimizer.optimize_quantization(
            train_loader, val_loader, test_data, test_labels
        )
        
        # 保存量化结果
        save_dir = self.config.get('output', {}).get('quantization_dir', './output/quantization')
        os.makedirs(save_dir, exist_ok=True)
        
        from quantization.model_quantization import save_quantization_results
        save_quantization_results(optimizer, save_dir)
        
        # 绘制优化结果
        optimizer.plot_optimization_results("quantization_optimization.png")
        
        self.logger.info(f"最佳量化策略: {results['best_strategy']}")
        self.logger.info("模型量化完成")
        
        return results[results['best_strategy']]['model']
    
    def setup_caching(self):
        """设置缓存系统"""
        self.logger.info("设置缓存系统")
        
        cache_config = self.config.get('cache', {})
        cache_manager = ModelCache(
            cache_dir=cache_config.get('cache_dir', './model_cache'),
            max_cache_size_gb=cache_config.get('max_cache_size_gb', 2.0)
        )
        
        preloader = ModelPreloader(
            max_concurrent_loads=cache_config.get('max_concurrent_loads', 3)
        )
        
        return cache_manager, preloader
    
    def setup_deployment(self):
        """设置部署系统"""
        self.logger.info("设置部署系统")
        
        # 系统分析
        specs = SystemAnalyzer.get_system_specs()
        orchestrator = DeploymentOrchestrator()
        
        # 分析并推荐部署方案
        recommendation = orchestrator.analyze_and_recommend(specs)
        
        self.logger.info(f"推荐部署类型: {recommendation['system_analysis']['recommended_deployment']}")
        self.logger.info(f"优化建议: {recommendation['system_analysis']['optimization_suggestions']}")
        
        return orchestrator, recommendation
    
    def setup_monitoring(self):
        """设置监控系统"""
        self.logger.info("设置监控系统")
        
        monitoring_config = self.config.get('monitoring', {})
        
        memory_optimizer = MemoryOptimizer(
            memory_limit_gb=monitoring_config.get('memory_limit_gb', 4.0),
            gc_threshold=monitoring_config.get('gc_threshold', 0.8)
        )
        
        resource_monitor = ResourceMonitor(
            monitoring_interval=monitoring_config.get('monitoring_interval', 1.0)
        )
        
        # 添加告警回调
        def alert_callback(alert):
            self.logger.warning(f"资源告警: {alert}")
        
        resource_monitor.add_alert_callback(alert_callback)
        resource_monitor.start_monitoring()
        
        return memory_optimizer, resource_monitor
    
    def run_full_optimization(self, model_path: str, train_loader=None, 
                            val_loader=None, test_data=None, test_labels=None):
        """运行完整优化流程"""
        self.logger.info("开始完整模型优化流程")
        
        # 1. 加载模型
        self.logger.info("加载模型")
        model = torch.load(model_path, map_location='cpu')
        original_model = model
        
        # 2. 设置缓存和监控
        cache_manager, preloader = self.setup_caching()
        memory_optimizer, resource_monitor = self.setup_monitoring()
        
        # 3. 模型剪枝
        if self.config.get('pruning', {}).get('enabled', False):
            model = self.run_pruning(model, train_loader, test_data, test_labels)
        
        # 4. 知识蒸馏
        if self.config.get('distillation', {}).get('enabled', False):
            # 这里需要教师模型和学生模型
            teacher_model = original_model  # 假设原始模型作为教师模型
            student_model = model  # 当前模型作为学生模型
            model = self.run_distillation(teacher_model, student_model, train_loader, val_loader)
        
        # 5. 模型量化
        if self.config.get('quantization', {}).get('enabled', False):
            model = self.run_quantization(model, train_loader, val_loader, test_data, test_labels)
        
        # 6. 部署分析
        orchestrator, recommendation = self.setup_deployment()
        
        # 7. 保存优化后的模型
        output_dir = self.config.get('output', {}).get('models_dir', './output/models')
        os.makedirs(output_dir, exist_ok=True)
        
        optimized_model_path = os.path.join(output_dir, 'optimized_model.pth')
        torch.save(model, optimized_model_path)
        
        # 8. 生成部署配置
        deployment_config = recommendation['recommended_config']
        config_path = os.path.join(output_dir, 'deployment_config.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(deployment_config, f, default_flow_style=False)
        
        # 9. 生成优化报告
        self.generate_optimization_report(
            original_model, model, cache_manager, resource_monitor, recommendation
        )
        
        self.logger.info("完整优化流程完成")
        return model, deployment_config
    
    def generate_optimization_report(self, original_model, optimized_model, 
                                   cache_manager, resource_monitor, recommendation):
        """生成优化报告"""
        self.logger.info("生成优化报告")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_summary': {
                'original_model_size_mb': self._calculate_model_size(original_model),
                'optimized_model_size_mb': self._calculate_model_size(optimized_model),
                'size_reduction_percent': self._calculate_size_reduction(original_model, optimized_model)
            },
            'cache_statistics': cache_manager.get_cache_stats(),
            'resource_monitoring': resource_monitor.get_metrics_summary(duration_minutes=1),
            'deployment_recommendation': recommendation,
            'optimization_config': self.config
        }
        
        # 保存报告
        report_path = self.config.get('output', {}).get('report_path', './output/optimization_report.json')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"优化报告已保存: {report_path}")
    
    def _calculate_model_size(self, model) -> float:
        """计算模型大小(MB)"""
        import torch
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024 / 1024
        return size_all_mb
    
    def _calculate_size_reduction(self, original_model, optimized_model) -> float:
        """计算模型大小减少百分比"""
        original_size = self._calculate_model_size(original_model)
        optimized_size = self._calculate_model_size(optimized_model)
        
        if original_size > 0:
            reduction_percent = ((original_size - optimized_size) / original_size) * 100
            return reduction_percent
        return 0.0


def create_sample_config():
    """创建示例配置文件"""
    config = {
        'logging': {
            'level': 'INFO'
        },
        'pruning': {
            'enabled': True,
            'type': 'unstructured',  # 'structured', 'unstructured', 'progressive'
            'ratio': 0.5,
            'fine_tune_epochs': 10
        },
        'distillation': {
            'enabled': False,
            'type': 'standard',  # 'standard', 'progressive'
            'temperature': 4.0,
            'alpha': 0.7,
            'beta': 0.3,
            'epochs': 50,
            'learning_rate': 0.001
        },
        'quantization': {
            'enabled': True
        },
        'cache': {
            'cache_dir': './model_cache',
            'max_cache_size_gb': 2.0,
            'max_concurrent_loads': 3
        },
        'monitoring': {
            'memory_limit_gb': 4.0,
            'gc_threshold': 0.8,
            'monitoring_interval': 1.0
        },
        'output': {
            'models_dir': './output/models',
            'pruning_dir': './output/pruning',
            'distillation_dir': './output/distillation',
            'quantization_dir': './output/quantization',
            'report_path': './output/optimization_report.json'
        }
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description='模型压缩和部署优化工具')
    parser.add_argument('--config', type=str, default='optimization_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--create_config', action='store_true',
                       help='创建示例配置文件')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='运行完整优化流程')
    
    args = parser.parse_args()
    
    # 创建示例配置
    if args.create_config:
        config = create_sample_config()
        with open(args.config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"示例配置文件已创建: {args.config}")
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 创建优化流水线
    pipeline = ModelOptimizationPipeline(args.config)
    
    if args.full_pipeline:
        # 运行完整优化流程
        try:
            optimized_model, deployment_config = pipeline.run_full_optimization(args.model_path)
            print("优化完成!")
            print(f"优化后的模型已保存到: ./output/models/optimized_model.pth")
            print(f"部署配置已保存到: ./output/models/deployment_config.yaml")
        except Exception as e:
            print(f"优化过程出错: {e}")
            pipeline.logger.error(f"优化过程出错: {e}")
    else:
        print("请使用 --full_pipeline 参数运行完整优化流程")
        print("或使用 --create_config 创建示例配置文件")


if __name__ == "__main__":
    main()