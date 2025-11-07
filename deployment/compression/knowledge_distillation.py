"""
知识蒸馏模块
实现教师-学生模型的知识转移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import copy
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class KnowledgeDistiller:
    """知识蒸馏器"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.7, beta: float = 0.3):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.beta = beta    # 任务损失权重
        self.distillation_history = []
        
    def compute_distillation_loss(self, student_logits: torch.Tensor, 
                                teacher_logits: torch.Tensor) -> torch.Tensor:
        """计算蒸馏损失"""
        # 使用KL散度计算软标签损失
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distillation_loss = F.kl_div(
            student_soft, teacher_soft, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return distillation_loss
    
    def compute_task_loss(self, student_logits: torch.Tensor, 
                         hard_labels: torch.Tensor) -> torch.Tensor:
        """计算任务损失"""
        return F.mse_loss(student_logits, hard_labels)
    
    def distill(self, train_loader, val_loader=None, epochs: int = 50, 
               lr: float = 0.001, device: str = 'cpu') -> nn.Module:
        """执行知识蒸馏"""
        print(f"开始知识蒸馏，温度: {self.temperature}, α: {self.alpha}, β: {self.beta}")
        
        # 移动模型到设备
        self.teacher_model = self.teacher_model.to(device)
        self.student_model = self.student_model.to(device)
        
        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # 优化器
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 评估初始性能
        initial_metrics = self.evaluate_model(self.student_model, val_loader, device)
        print(f"蒸馏前学生模型性能: {initial_metrics}")
        
        best_student = copy.deepcopy(self.student_model)
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练阶段
            self.student_model.train()
            total_distillation_loss = 0
            total_task_loss = 0
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)
                
                student_logits = self.student_model(data)
                
                # 计算损失
                distillation_loss = self.compute_distillation_loss(student_logits, teacher_logits)
                task_loss = self.compute_task_loss(student_logits, target)
                
                loss = self.alpha * distillation_loss + self.beta * task_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_distillation_loss += distillation_loss.item()
                total_task_loss += task_loss.item()
                total_loss += loss.item()
            
            scheduler.step()
            
            # 评估阶段
            if val_loader is not None and (epoch + 1) % 10 == 0:
                metrics = self.evaluate_model(self.student_model, val_loader, device)
                
                # 记录历史
                self.distillation_history.append({
                    'epoch': epoch + 1,
                    'distillation_loss': total_distillation_loss / len(train_loader),
                    'task_loss': total_task_loss / len(train_loader),
                    'total_loss': total_loss / len(train_loader),
                    'metrics': metrics
                })
                
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  蒸馏损失: {total_distillation_loss / len(train_loader):.6f}")
                print(f"  任务损失: {total_task_loss / len(train_loader):.6f}")
                print(f"  总损失: {total_loss / len(train_loader):.6f}")
                print(f"  验证性能: {metrics}")
                
                # 保存最佳模型
                if metrics.get('mse', float('inf')) < best_loss:
                    best_loss = metrics.get('mse', float('inf'))
                    best_student = copy.deepcopy(self.student_model)
            else:
                # 记录历史（无验证）
                self.distillation_history.append({
                    'epoch': epoch + 1,
                    'distillation_loss': total_distillation_loss / len(train_loader),
                    'task_loss': total_task_loss / len(train_loader),
                    'total_loss': total_loss / len(train_loader)
                })
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: 总损失: {total_loss / len(train_loader):.6f}")
        
        print("知识蒸馏完成")
        return best_student
    
    def evaluate_model(self, model: nn.Module, data_loader, device: str) -> Dict:
        """评估模型性能"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                predictions = model(data)
                loss = criterion(predictions, target)
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'loss': total_loss / len(data_loader)
        }


class ProgressiveDistiller:
    """渐进式知识蒸馏器"""
    
    def __init__(self, teacher_models: List[nn.Module], student_model: nn.Module,
                 temperatures: List[float] = [1.0, 2.0, 4.0, 8.0],
                 alphas: List[float] = [0.9, 0.7, 0.5, 0.3]):
        self.teacher_models = teacher_models
        self.student_model = student_model
        self.temperatures = temperatures
        self.alphas = alphas
        self.distillation_history = []
        
    def progressive_distill(self, train_loader, val_loader=None, epochs_per_stage: int = 25,
                          lr: float = 0.001, device: str = 'cpu') -> nn.Module:
        """渐进式蒸馏"""
        print("开始渐进式知识蒸馏")
        
        current_student = copy.deepcopy(self.student_model)
        
        for stage, (teacher, temperature, alpha) in enumerate(zip(
            self.teacher_models, self.temperatures, self.alphas)):
            
            print(f"\n=== 阶段 {stage+1}/{len(self.teacher_models)} ===")
            print(f"教师模型 {stage+1}, 温度: {temperature}, α: {alpha}")
            
            # 创建蒸馏器
            distiller = KnowledgeDistiller(
                teacher, current_student, 
                temperature=temperature, alpha=alpha, beta=1.0-alpha
            )
            
            # 执行蒸馏
            current_student = distiller.distill(
                train_loader, val_loader, epochs_per_stage, lr, device
            )
            
            # 记录阶段结果
            stage_metrics = {}
            if val_loader is not None:
                stage_metrics = distiller.evaluate_model(current_student, val_loader, device)
            
            self.distillation_history.append({
                'stage': stage + 1,
                'temperature': temperature,
                'alpha': alpha,
                'metrics': stage_metrics,
                'history': distiller.distillation_history
            })
        
        print("渐进式知识蒸馏完成")
        return current_student


class FeatureDistiller:
    """特征蒸馏器 - 蒸馏中间层特征"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 feature_layers: List[str], alpha: float = 0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.feature_layers = feature_layers
        self.alpha = alpha
        self.feature_distillation_history = []
        
    def extract_features(self, model: nn.Module, x: torch.Tensor, 
                        target_layers: List[str]) -> Dict[str, torch.Tensor]:
        """提取特征"""
        features = {}
        
        def hook_fn(module, input, output):
            for layer_name in target_layers:
                if layer_name in str(module):
                    features[layer_name] = output
        
        # 注册钩子
        hooks = []
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in target_layers):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            _ = model(x)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return features
    
    def compute_feature_loss(self, student_features: torch.Tensor, 
                           teacher_features: torch.Tensor) -> torch.Tensor:
        """计算特征损失"""
        # 使用MSE损失
        return F.mse_loss(student_features, teacher_features)
    
    def distill_features(self, train_loader, epochs: int = 30, 
                        lr: float = 0.001, device: str = 'cpu') -> nn.Module:
        """执行特征蒸馏"""
        print(f"开始特征蒸馏，层: {self.feature_layers}")
        
        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # 优化器
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # 提取特征
                teacher_features = self.extract_features(self.teacher_model, data, self.feature_layers)
                student_features = self.extract_features(self.student_model, data, self.feature_layers)
                
                # 计算特征损失
                feature_loss = 0
                for layer_name in self.feature_layers:
                    if layer_name in teacher_features and layer_name in student_features:
                        feature_loss += self.compute_feature_loss(
                            student_features[layer_name], teacher_features[layer_name]
                        )
                
                # 计算任务损失
                student_logits = self.student_model(data)
                task_loss = F.mse_loss(student_logits, target)
                
                # 总损失
                loss = self.alpha * feature_loss + (1 - self.alpha) * task_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print("特征蒸馏完成")
        return self.student_model


def save_distillation_results(distiller, student_model, save_dir: str):
    """保存蒸馏结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存学生模型
    torch.save(student_model.state_dict(), os.path.join(save_dir, "distilled_student.pth"))
    
    # 保存蒸馏信息
    distillation_info = {
        'temperature': getattr(distiller, 'temperature', None),
        'alpha': getattr(distiller, 'alpha', None),
        'beta': getattr(distiller, 'beta', None),
        'distillation_history': getattr(distiller, 'distillation_history', [])
    }
    
    with open(os.path.join(save_dir, "distillation_info.json"), 'w') as f:
        json.dump(distillation_info, f, indent=2)
    
    print(f"蒸馏结果已保存到: {save_dir}")


def plot_distillation_history(history: List[Dict], save_path: str = "distillation_history.png"):
    """绘制蒸馏历史"""
    if not history:
        print("没有蒸馏历史记录")
        return
    
    epochs = [h['epoch'] for h in history]
    distillation_losses = [h.get('distillation_loss', 0) for h in history]
    task_losses = [h.get('task_loss', 0) for h in history]
    total_losses = [h.get('total_loss', 0) for h in history]
    
    # 提取验证指标
    metrics_epochs = []
    mse_scores = []
    mae_scores = []
    
    for h in history:
        if 'metrics' in h and h['metrics']:
            metrics_epochs.append(h['epoch'])
            mse_scores.append(h['metrics'].get('mse', 0))
            mae_scores.append(h['metrics'].get('mae', 0))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    ax1.plot(epochs, distillation_losses, 'b-', label='蒸馏损失')
    ax1.plot(epochs, task_losses, 'r-', label='任务损失')
    ax1.plot(epochs, total_losses, 'g-', label='总损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('蒸馏损失曲线')
    ax1.legend()
    ax1.grid(True)
    
    # MSE曲线
    if metrics_epochs:
        ax2.plot(metrics_epochs, mse_scores, 'o-', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.set_title('验证MSE')
        ax2.grid(True)
    
    # MAE曲线
    if metrics_epochs:
        ax3.plot(metrics_epochs, mae_scores, 's-', color='purple')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MAE')
        ax3.set_title('验证MAE')
        ax3.grid(True)
    
    # 学习率曲线（如果有）
    if 'learning_rate' in history[0]:
        lrs = [h.get('learning_rate', 0) for h in history]
        ax4.plot(epochs, lrs, 'c-', label='学习率')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('学习率调度')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"蒸馏历史图已保存到: {save_path}")


if __name__ == "__main__":
    # 示例使用
    class TeacherModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 30)
            self.fc3 = nn.Linear(30, 1)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    class StudentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 1)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    # 创建模型
    teacher = TeacherModel()
    student = StudentModel()
    
    # 创建虚拟数据
    train_data = torch.randn(1000, 10)
    train_labels = torch.randn(1000, 1)
    val_data = torch.randn(200, 10)
    val_labels = torch.randn(200, 1)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # 测试知识蒸馏
    print("=== 知识蒸馏测试 ===")
    distiller = KnowledgeDistiller(teacher, student, temperature=4.0, alpha=0.7)
    distilled_student = distiller.distill(train_loader, val_loader, epochs=20, device='cpu')
    
    print("知识蒸馏测试完成")