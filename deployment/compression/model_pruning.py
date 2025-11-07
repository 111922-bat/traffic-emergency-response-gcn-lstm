"""
模型剪枝模块
支持结构化剪枝和非结构化剪枝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


class StructuredPruner:
    """结构化剪枝器 - 剪除整个通道/层"""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.original_weights = {}
        self.pruned_weights = {}
        self.pruning_history = []
        
    def _get_layer_importance(self, layer: nn.Module) -> torch.Tensor:
        """计算层的重要性分数"""
        if isinstance(layer, nn.Linear):
            # 使用L2范数作为重要性指标
            importance = torch.norm(layer.weight, p=2, dim=1)
        elif isinstance(layer, nn.Conv2d):
            # 使用L2范数作为重要性指标
            importance = torch.norm(layer.weight.view(layer.weight.size(0), -1), p=2, dim=1)
        else:
            importance = torch.ones(layer.out_features if hasattr(layer, 'out_features') else 1)
        
        return importance
    
    def _prune_layer_channels(self, layer: nn.Module, importance: torch.Tensor, 
                            pruning_ratio: float) -> Tuple[nn.Module, torch.Tensor]:
        """剪除通道并返回新的层"""
        num_channels = importance.size(0)
        num_prune = int(num_channels * pruning_ratio)
        
        # 找到重要性最低的通道
        _, indices = torch.topk(importance, num_prune, largest=False)
        
        if isinstance(layer, nn.Linear):
            # 线性层剪枝
            new_weight = layer.weight.clone()
            new_bias = layer.bias.clone() if layer.bias is not None else None
            
            # 移除低重要性的输出通道
            keep_indices = torch.tensor([i for i in range(num_channels) if i not in indices])
            new_weight = new_weight[keep_indices]
            if new_bias is not None:
                new_bias = new_bias[keep_indices]
            
            # 创建新层
            new_layer = nn.Linear(layer.in_features, len(keep_indices))
            new_layer.weight.data = new_weight
            if new_bias is not None:
                new_layer.bias.data = new_bias
            
        elif isinstance(layer, nn.Conv2d):
            # 卷积层剪枝
            new_weight = layer.weight.clone()
            new_bias = layer.bias.clone() if layer.bias is not None else None
            
            # 移除低重要性的输出通道
            keep_indices = torch.tensor([i for i in range(num_channels) if i not in indices])
            new_weight = new_weight[keep_indices]
            if new_bias is not None:
                new_bias = new_bias[keep_indices]
            
            # 创建新层
            new_layer = nn.Conv2d(
                layer.in_channels, len(keep_indices), 
                layer.kernel_size, layer.stride, layer.padding,
                layer.dilation, layer.groups, layer.bias is not None
            )
            new_layer.weight.data = new_weight
            if new_bias is not None:
                new_layer.bias.data = new_bias
        else:
            new_layer = copy.deepcopy(layer)
            
        return new_layer, importance[keep_indices]
    
    def prune_model(self, test_data: torch.Tensor = None, test_labels: torch.Tensor = None) -> nn.Module:
        """执行模型剪枝"""
        print(f"开始结构化剪枝，剪枝比例: {self.pruning_ratio}")
        
        # 保存原始模型
        self.original_weights = {}
        for name, param in self.model.named_parameters():
            self.original_weights[name] = param.clone()
        
        # 创建新模型
        new_model = copy.deepcopy(self.model)
        
        # 记录剪枝前的性能
        original_accuracy = None
        if test_data is not None and test_labels is not None:
            original_accuracy = self._evaluate_model(self.model, test_data, test_labels)
            print(f"剪枝前准确率: {original_accuracy:.4f}")
        
        # 逐层剪枝
        layer_importance = {}
        for name, module in new_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                print(f"剪枝层: {name}")
                importance = self._get_layer_importance(module)
                new_module, new_importance = self._prune_layer_channels(
                    module, importance, self.pruning_ratio
                )
                layer_importance[name] = new_importance
                
                # 替换层
                self._replace_layer(new_model, name, new_module)
        
        # 记录剪枝后的性能
        pruned_accuracy = None
        if test_data is not None and test_labels is not None:
            pruned_accuracy = self._evaluate_model(new_model, test_data, test_labels)
            print(f"剪枝后准确率: {pruned_accuracy:.4f}")
        
        # 记录剪枝历史
        self.pruning_history.append({
            'pruning_ratio': self.pruning_ratio,
            'original_accuracy': original_accuracy,
            'pruned_accuracy': pruned_accuracy,
            'layer_importance': {k: v.tolist() for k, v in layer_importance.items()}
        })
        
        return new_model
    
    def _replace_layer(self, model: nn.Module, layer_name: str, new_layer: nn.Module):
        """替换模型中的层"""
        parts = layer_name.split('.')
        current = model
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], new_layer)
    
    def _evaluate_model(self, model: nn.Module, test_data: torch.Tensor, test_labels: torch.Tensor) -> float:
        """评估模型性能"""
        model.eval()
        with torch.no_grad():
            outputs = model(test_data)
            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(test_labels.cpu().numpy(), predicted.cpu().numpy())
            else:
                # 回归任务
                mse = F.mse_loss(outputs.squeeze(), test_labels.float())
                accuracy = 1.0 / (1.0 + mse.item())  # 转换为准确率指标
        return accuracy


class UnstructuredPruner:
    """非结构化剪枝器 - 剪除单个权重"""
    
    def __init__(self, model: nn.Module, sparsity_ratio: float = 0.5):
        self.model = model
        self.sparsity_ratio = sparsity_ratio
        self.masks = {}
        self.original_weights = {}
        
    def create_masks(self) -> Dict[str, torch.Tensor]:
        """创建剪枝掩码"""
        masks = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # 计算阈值
                threshold = torch.quantile(torch.abs(param.data), self.sparsity_ratio)
                # 创建掩码
                mask = (torch.abs(param.data) > threshold).float()
                masks[name] = mask
        return masks
    
    def apply_masks(self, model: nn.Module, masks: Dict[str, torch.Tensor]) -> nn.Module:
        """应用剪枝掩码"""
        for name, param in model.named_parameters():
            if name in masks:
                param.data *= masks[name]
        return model
    
    def prune_model(self) -> nn.Module:
        """执行非结构化剪枝"""
        print(f"开始非结构化剪枝，稀疏度: {self.sparsity_ratio}")
        
        # 保存原始权重
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.original_weights[name] = param.data.clone()
        
        # 创建并应用掩码
        masks = self.create_masks()
        self.masks = masks
        
        # 应用掩码到模型
        self.apply_masks(self.model, masks)
        
        # 计算实际稀疏度
        total_weights = sum(p.numel() for name, p in self.model.named_parameters() if 'weight' in name)
        zero_weights = sum((p == 0).sum().item() for name, p in self.model.named_parameters() if 'weight' in name)
        actual_sparsity = zero_weights / total_weights
        
        print(f"实际稀疏度: {actual_sparsity:.4f}")
        
        return self.model
    
    def fine_tune(self, train_loader, epochs: int = 10, lr: float = 0.001):
        """微调剪枝后的模型"""
        print(f"开始微调，轮数: {epochs}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()  # 根据任务调整损失函数
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 应用掩码
                self.apply_masks(self.model, self.masks)
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print("微调完成")


class PruningScheduler:
    """剪枝调度器 - 渐进式剪枝"""
    
    def __init__(self, model: nn.Module, initial_sparsity: float = 0.1, 
                 final_sparsity: float = 0.7, steps: int = 10):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.steps = steps
        self.pruning_history = []
        
    def progressive_prune(self, train_loader, test_data: torch.Tensor = None, 
                         test_labels: torch.Tensor = None) -> nn.Module:
        """渐进式剪枝"""
        print("开始渐进式剪枝")
        
        current_model = copy.deepcopy(self.model)
        
        for step in range(self.steps):
            # 计算当前步骤的剪枝比例
            progress = step / (self.steps - 1)
            current_sparsity = self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
            
            print(f"步骤 {step+1}/{self.steps}, 稀疏度: {current_sparsity:.4f}")
            
            # 执行剪枝
            pruner = UnstructuredPruner(current_model, current_sparsity)
            current_model = pruner.prune_model()
            
            # 微调
            pruner.fine_tune(train_loader, epochs=5)
            
            # 评估性能
            accuracy = None
            if test_data is not None and test_labels is not None:
                accuracy = self._evaluate_model(current_model, test_data, test_labels)
            
            # 记录历史
            self.pruning_history.append({
                'step': step + 1,
                'sparsity': current_sparsity,
                'accuracy': accuracy
            })
        
        return current_model
    
    def _evaluate_model(self, model: nn.Module, test_data: torch.Tensor, test_labels: torch.Tensor) -> float:
        """评估模型性能"""
        model.eval()
        with torch.no_grad():
            outputs = model(test_data)
            mse = F.mse_loss(outputs.squeeze(), test_labels.float())
            accuracy = 1.0 / (1.0 + mse.item())
        return accuracy
    
    def plot_pruning_history(self, save_path: str = "pruning_history.png"):
        """绘制剪枝历史"""
        if not self.pruning_history:
            print("没有剪枝历史记录")
            return
        
        steps = [h['step'] for h in self.pruning_history]
        sparsities = [h['sparsity'] for h in self.pruning_history]
        accuracies = [h['accuracy'] for h in self.pruning_history if h['accuracy'] is not None]
        accuracy_steps = [h['step'] for h in self.pruning_history if h['accuracy'] is not None]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 稀疏度曲线
        ax1.plot(steps, sparsities, 'b-o', label='稀疏度')
        ax1.set_xlabel('步骤')
        ax1.set_ylabel('稀疏度')
        ax1.set_title('渐进式剪枝稀疏度')
        ax1.grid(True)
        ax1.legend()
        
        # 准确率曲线
        if accuracies:
            ax2.plot(accuracy_steps, accuracies, 'r-o', label='准确率')
            ax2.set_xlabel('步骤')
            ax2.set_ylabel('准确率')
            ax2.set_title('剪枝过程中准确率变化')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"剪枝历史图已保存到: {save_path}")


def save_pruning_results(pruner, model, save_dir: str):
    """保存剪枝结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, "pruned_model.pth"))
    
    # 保存剪枝信息
    pruning_info = {
        'pruning_type': type(pruner).__name__,
        'pruning_ratio': getattr(pruner, 'pruning_ratio', None),
        'sparsity_ratio': getattr(pruner, 'sparsity_ratio', None),
        'pruning_history': pruner.pruning_history if hasattr(pruner, 'pruning_history') else []
    }
    
    with open(os.path.join(save_dir, "pruning_info.json"), 'w') as f:
        json.dump(pruning_info, f, indent=2)
    
    print(f"剪枝结果已保存到: {save_dir}")


if __name__ == "__main__":
    # 示例使用
    # 创建简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 15)
            self.fc3 = nn.Linear(15, 1)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    # 创建模型和数据
    model = SimpleModel()
    test_data = torch.randn(100, 10)
    test_labels = torch.randn(100, 1)
    
    # 测试结构化剪枝
    print("=== 结构化剪枝测试 ===")
    structured_pruner = StructuredPruner(model, pruning_ratio=0.3)
    pruned_model = structured_pruner.prune_model(test_data, test_labels)
    
    # 测试非结构化剪枝
    print("\n=== 非结构化剪枝测试 ===")
    unstructured_pruner = UnstructuredPruner(model, sparsity_ratio=0.5)
    pruned_model = unstructured_pruner.prune_model()
    
    # 测试渐进式剪枝
    print("\n=== 渐进式剪枝测试 ===")
    scheduler = PruningScheduler(model, initial_sparsity=0.1, final_sparsity=0.6, steps=5)
    final_model = scheduler.progressive_prune(None, test_data, test_labels)
    
    print("模型剪枝测试完成")