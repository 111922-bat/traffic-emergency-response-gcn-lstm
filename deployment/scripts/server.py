#!/usr/bin/env python3
"""
优化的模型服务示例
支持缓存、内存优化和资源监控
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

# 添加部署模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
import psutil
import yaml

# 导入优化模块
from caching.model_cache import ModelCache, AdaptiveCacheStrategy
from optimization.memory_optimization import MemoryOptimizer, ResourceMonitor, AdaptiveBatchSizer


class OptimizedModelService:
    """优化的模型服务"""
    
    def __init__(self, config):
        self.config = config
        self.app = Flask(__name__)
        self.setup_logging()
        self.setup_components()
        self.setup_routes()
        
    def setup_logging(self):
        """设置日志"""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_components(self):
        """设置组件"""
        # 内存优化器
        memory_config = self.config.get('memory', {})
        self.memory_optimizer = MemoryOptimizer(
            memory_limit_gb=memory_config.get('memory_limit_gb', 4.0),
            gc_threshold=memory_config.get('gc_threshold', 0.8)
        )
        
        # 资源监控器
        monitor_config = self.config.get('monitor', {})
        self.resource_monitor = ResourceMonitor(
            monitoring_interval=monitor_config.get('monitoring_interval', 1.0)
        )
        self.resource_monitor.start_monitoring()
        
        # 缓存管理器
        cache_config = self.config.get('cache', {})
        self.cache_manager = ModelCache(
            cache_dir=cache_config.get('cache_dir', './model_cache'),
            max_cache_size_gb=cache_config.get('max_cache_size_gb', 2.0)
        )
        
        # 自适应缓存策略
        self.adaptive_cache = AdaptiveCacheStrategy(self.cache_manager)
        
        # 自适应批处理大小调整器
        self.batch_sizer = AdaptiveBatchSizer(
            initial_batch_size=self.config.get('batch_size', 32),
            min_batch_size=1,
            max_batch_size=256
        )
        
        # 加载模型
        self.model = None
        self.model_loaded = False
        self.load_model()
        
    def load_model(self):
        """加载模型"""
        model_path = self.config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            self.logger.error(f"模型文件不存在: {model_path}")
            return
            
        try:
            self.logger.info(f"加载模型: {model_path}")
            self.model = torch.load(model_path, map_location='cpu')
            self.model.eval()
            
            # 优化模型内存
            self.model = self.memory_optimizer.optimize_model_memory(self.model)
            
            self.model_loaded = True
            self.logger.info("模型加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.model_loaded = False
    
    def setup_routes(self):
        """设置路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'model_loaded': self.model_loaded,
                'memory_usage': self.get_memory_stats(),
                'resource_usage': self.get_resource_stats()
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """预测接口"""
            if not self.model_loaded:
                return jsonify({'error': '模型未加载'}), 503
            
            try:
                # 获取输入数据
                data = request.get_json()
                if not data or 'input' not in data:
                    return jsonify({'error': '缺少输入数据'}), 400
                
                input_data = torch.tensor(data['input'], dtype=torch.float32)
                
                # 自适应批处理
                batch_size = self.batch_sizer.get_recommended_batch_size()
                
                # 预测
                start_time = time.time()
                with torch.no_grad():
                    if input_data.dim() == 1:
                        input_data = input_data.unsqueeze(0)
                    
                    # 分批处理
                    outputs = []
                    for i in range(0, len(input_data), batch_size):
                        batch = input_data[i:i+batch_size]
                        batch_output = self.model(batch)
                        outputs.append(batch_output)
                    
                    final_output = torch.cat(outputs, dim=0)
                
                inference_time = time.time() - start_time
                
                # 记录访问模式
                self.adaptive_cache.record_access('model', time.time())
                
                # 调整批处理大小
                memory_usage = psutil.virtual_memory().percent
                self.batch_sizer.adjust_batch_size(inference_time, memory_usage)
                
                return jsonify({
                    'prediction': final_output.cpu().numpy().tolist(),
                    'inference_time': inference_time,
                    'batch_size_used': batch_size,
                    'memory_usage': memory_usage
                })
                
            except Exception as e:
                self.logger.error(f"预测失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/cache/stats', methods=['GET'])
        def cache_stats():
            """缓存统计"""
            return jsonify(self.cache_manager.get_cache_stats())
        
        @self.app.route('/resource/stats', methods=['GET'])
        def resource_stats():
            """资源统计"""
            return jsonify({
                'memory': self.get_memory_stats(),
                'resource': self.get_resource_stats(),
                'batch_size': self.batch_sizer.get_recommended_batch_size()
            })
        
        @self.app.route('/optimize', methods=['POST'])
        def optimize():
            """手动优化"""
            try:
                # 垃圾回收
                self.memory_optimizer.smart_gc(force=True)
                
                # 获取优化建议
                recommendations = self.memory_optimizer.get_optimization_recommendations()
                
                return jsonify({
                    'status': 'optimized',
                    'recommendations': recommendations,
                    'memory_stats': self.get_memory_stats()
                })
                
            except Exception as e:
                self.logger.error(f"优化失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/reload', methods=['POST'])
        def reload_model():
            """重新加载模型"""
            try:
                self.load_model()
                return jsonify({
                    'status': 'reloaded',
                    'model_loaded': self.model_loaded
                })
            except Exception as e:
                self.logger.error(f"重新加载失败: {e}")
                return jsonify({'error': str(e)}), 500
    
    def get_memory_stats(self):
        """获取内存统计"""
        return self.memory_optimizer.monitor_memory()
    
    def get_resource_stats(self):
        """获取资源统计"""
        current_metrics = self.resource_monitor.get_current_metrics()
        if current_metrics:
            return {
                'cpu_percent': current_metrics.cpu_percent,
                'memory_percent': current_metrics.memory_percent,
                'gpu_percent': current_metrics.gpu_percent
            }
        return {}
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """运行服务"""
        self.logger.info(f"启动优化模型服务: {host}:{port}")
        self.logger.info(f"模型路径: {self.config.get('model_path')}")
        self.logger.info(f"批处理大小: {self.config.get('batch_size')}")
        self.logger.info(f"启用GPU: {self.config.get('enable_gpu', False)}")
        self.logger.info(f"启用缓存: {self.config.get('enable_caching', True)}")
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            self.logger.info("接收到中断信号，正在停止服务...")
        finally:
            self.resource_monitor.stop_monitoring()
            self.logger.info("服务已停止")


def load_config(config_path):
    """加载配置"""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def create_default_config():
    """创建默认配置"""
    return {
        'model_path': './models/best_model.pth',
        'port': 8080,
        'batch_size': 32,
        'max_memory_mb': 2048,
        'enable_gpu': False,
        'enable_caching': True,
        'cache_size_mb': 512,
        'log_level': 'INFO',
        'memory': {
            'memory_limit_gb': 4.0,
            'gc_threshold': 0.8
        },
        'monitor': {
            'monitoring_interval': 1.0
        },
        'cache': {
            'cache_dir': './model_cache',
            'max_cache_size_gb': 2.0
        }
    }


def main():
    parser = argparse.ArgumentParser(description='优化的模型服务')
    parser.add_argument('--model_path', type=str, help='模型文件路径')
    parser.add_argument('--port', type=int, default=8080, help='服务端口')
    parser.add_argument('--workers', type=int, default=4, help='工作进程数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--max_memory_mb', type=int, default=2048, help='最大内存使用(MB)')
    parser.add_argument('--enable_gpu', action='store_true', help='启用GPU支持')
    parser.add_argument('--enable_caching', action='store_true', help='启用缓存')
    parser.add_argument('--cache_size_mb', type=int, default=512, help='缓存大小(MB)')
    parser.add_argument('--log_level', type=str, default='INFO', help='日志级别')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # 命令行参数覆盖配置
    if args.model_path:
        config['model_path'] = args.model_path
    if args.port:
        config['port'] = args.port
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_memory_mb:
        config['max_memory_mb'] = args.max_memory_mb
    if args.enable_gpu:
        config['enable_gpu'] = True
    if args.enable_caching:
        config['enable_caching'] = True
    if args.cache_size_mb:
        config['cache_size_mb'] = args.cache_size_mb
    if args.log_level:
        config['log_level'] = args.log_level
    
    # 创建并运行服务
    service = OptimizedModelService(config)
    service.run(port=config['port'], debug=args.debug)


if __name__ == '__main__':
    main()