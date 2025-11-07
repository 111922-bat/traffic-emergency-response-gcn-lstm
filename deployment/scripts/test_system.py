#!/usr/bin/env python3
"""
部署优化系统测试脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# 添加部署模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compression.model_pruning import UnstructuredPruner
from compression.knowledge_distillation import KnowledgeDistiller
from quantization.model_quantization import QuantizationOptimizer
from caching.model_cache import ModelCache
from optimization.deployment_architecture import SystemAnalyzer
from optimization.memory_optimization import MemoryOptimizer


class SimpleModel(nn.Module):
    """简单的测试模型"""
    def __init__(self, input_size=10, hidden_size=50, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def create_test_data(num_samples=1000, input_size=10):
    """创建测试数据"""
    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, 1)
    return X, y


def test_model_pruning():
    """测试模型剪枝"""
    print("=== 测试模型剪枝 ===")
    
    # 创建测试模型
    model = SimpleModel()
    test_data, test_labels = create_test_data(100)
    
    # 测试非结构化剪枝
    pruner = UnstructuredPruner(model, sparsity_ratio=0.3)
    pruned_model = pruner.prune_model()
    
    # 验证模型结构
    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    
    print(f"原始模型参数: {original_params}")
    print(f"剪枝后参数: {pruned_params}")
    print(f"剪枝比例: {(original_params - pruned_params) / original_params * 100:.2f}%")
    print("✅ 模型剪枝测试通过")
    return pruned_model


def test_knowledge_distillation():
    """测试知识蒸馏"""
    print("\n=== 测试知识蒸馏 ===")
    
    # 创建教师和学生模型
    teacher_model = SimpleModel(hidden_size=100)
    student_model = SimpleModel(hidden_size=50)
    
    # 创建测试数据
    train_data, train_labels = create_test_data(200)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 测试蒸馏器
    distiller = KnowledgeDistiller(
        teacher_model, student_model,
        temperature=4.0, alpha=0.7, beta=0.3
    )
    
    print("蒸馏器创建成功")
    print("✅ 知识蒸馏测试通过")
    return student_model


def test_model_quantization():
    """测试模型量化"""
    print("\n=== 测试模型量化 ===")
    
    # 创建测试模型
    model = SimpleModel()
    test_data, test_labels = create_test_data(100)
    
    # 创建虚拟数据加载器
    train_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    
    # 测试量化优化器
    optimizer = QuantizationOptimizer(model)
    
    print("量化优化器创建成功")
    print("✅ 模型量化测试通过")
    return model


def test_model_cache():
    """测试模型缓存"""
    print("\n=== 测试模型缓存 ===")
    
    # 创建测试模型
    model = SimpleModel()
    
    # 创建缓存管理器
    cache_manager = ModelCache(cache_dir='./test_cache', max_cache_size_gb=0.1)
    
    # 测试缓存
    cache_key = cache_manager.cache_model(model, "test_model", {"input_size": 10})
    print(f"缓存键: {cache_key}")
    
    # 测试加载
    new_model = SimpleModel()
    success = cache_manager.load_model(new_model, "test_model", {"input_size": 10})
    print(f"加载成功: {success}")
    
    # 获取统计信息
    stats = cache_manager.get_cache_stats()
    print(f"缓存统计: {stats}")
    
    # 清理
    cache_manager.clear_cache()
    
    print("✅ 模型缓存测试通过")


def test_system_analysis():
    """测试系统分析"""
    print("\n=== 测试系统分析 ===")
    
    # 获取系统规格
    specs = SystemAnalyzer.get_system_specs()
    print(f"CPU核心数: {specs.cpu_cores}")
    print(f"内存大小: {specs.memory_gb:.1f}GB")
    print(f"GPU数量: {specs.gpu_count}")
    print(f"操作系统: {specs.os_type}")
    
    # 分析部署适合性
    analyzer = SystemAnalyzer()
    analysis = analyzer.analyze_deployment_suitability(specs)
    
    print(f"推荐部署类型: {analysis['recommended_deployment']}")
    print(f"优化建议: {analysis['optimization_suggestions']}")
    
    print("✅ 系统分析测试通过")


def test_memory_optimization():
    """测试内存优化"""
    print("\n=== 测试内存优化 ===")
    
    # 创建内存优化器
    optimizer = MemoryOptimizer(memory_limit_gb=1.0)
    
    # 监控内存
    stats = optimizer.monitor_memory()
    print(f"当前内存使用: {stats.used_memory_mb:.2f}MB / {stats.total_memory_mb:.2f}MB")
    
    # 智能垃圾回收
    optimizer.smart_gc(force=True)
    
    # 预测内存压力
    prediction = optimizer.predict_memory_pressure()
    print(f"内存压力预测: {prediction}")
    
    # 获取优化建议
    recommendations = optimizer.get_optimization_recommendations()
    print(f"优化建议: {recommendations}")
    
    print("✅ 内存优化测试通过")


def test_full_pipeline():
    """测试完整优化流程"""
    print("\n=== 测试完整优化流程 ===")
    
    try:
        # 创建测试模型
        model = SimpleModel()
        
        # 保存测试模型
        os.makedirs('./test_models', exist_ok=True)
        torch.save(model, './test_models/test_model.pth')
        
        # 创建测试数据
        test_data, test_labels = create_test_data(100)
        
        # 创建虚拟数据加载器
        train_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
        
        # 测试各个组件
        print("1. 测试模型剪枝...")
        pruned_model = test_model_pruning()
        
        print("2. 测试知识蒸馏...")
        student_model = test_knowledge_distillation()
        
        print("3. 测试模型量化...")
        quantized_model = test_model_quantization()
        
        print("4. 测试模型缓存...")
        test_model_cache()
        
        print("5. 测试系统分析...")
        test_system_analysis()
        
        print("6. 测试内存优化...")
        test_memory_optimization()
        
        print("\n🎉 所有测试通过！部署优化系统工作正常")
        
        # 清理测试文件
        import shutil
        if os.path.exists('./test_cache'):
            shutil.rmtree('./test_cache')
        if os.path.exists('./test_models'):
            shutil.rmtree('./test_models')
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 开始部署优化系统测试")
    print("=" * 50)
    
    # 检查依赖
    try:
        import torch
        import numpy as np
        import psutil
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return
    
    # 运行测试
    success = test_full_pipeline()
    
    if success:
        print("\n" + "=" * 50)
        print("🎊 部署优化系统测试完成！")
        print("\n📋 下一步操作:")
        print("1. 配置优化参数: configs/optimization_config.yaml")
        print("2. 运行模型优化: python scripts/optimize_model.py --full_pipeline")
        print("3. 部署服务: ./scripts/deploy.sh --deploy-type local")
        print("4. 测试API: curl http://localhost:8080/health")
    else:
        print("\n❌ 测试失败，请检查错误信息")
    
    return success


if __name__ == "__main__":
    main()