#!/usr/bin/env python3
"""
依赖关系分析和可视化
Dependency Analysis and Visualization

分析重构前后的模块依赖关系，生成可视化图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


@dataclass
class ModuleDependency:
    """模块依赖信息"""
    module_name: str
    dependencies: List[str]
    dependents: List[str]
    file_path: str
    has_hardcoded_paths: bool
    error_handling_pattern: str


class DependencyAnalyzer:
    """依赖关系分析器"""
    
    def __init__(self):
        self.modules = {}
        self.dependency_graph = nx.DiGraph()
        self.hardcoded_paths = []
        self.error_patterns = {}
    
    def analyze_codebase(self, code_root: Path) -> Dict[str, ModuleDependency]:
        """分析代码库依赖关系"""
        logger.info(f"开始分析代码库: {code_root}")
        
        # 扫描Python文件
        python_files = list(code_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                self._analyze_file(file_path, code_root)
            except Exception as e:
                logger.warning(f"分析文件失败 {file_path}: {e}")
        
        logger.info(f"分析完成，发现 {len(self.modules)} 个模块")
        return self.modules
    
    def _analyze_file(self, file_path: Path, code_root: Path):
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 获取相对路径
            rel_path = file_path.relative_to(code_root)
            
            # 提取模块名
            module_name = self._extract_module_name(rel_path)
            
            # 分析依赖
            imports = self._extract_imports(content)
            hardcoded_paths = self._find_hardcoded_paths(content)
            error_patterns = self._analyze_error_patterns(content)
            
            # 创建依赖信息
            module_dep = ModuleDependency(
                module_name=module_name,
                dependencies=imports,
                dependents=[],  # 稍后填充
                file_path=str(rel_path),
                has_hardcoded_paths=len(hardcoded_paths) > 0,
                error_handling_pattern=error_patterns
            )
            
            self.modules[module_name] = module_dep
            self.hardcoded_paths.extend(hardcoded_paths)
            self.error_patterns[module_name] = error_patterns
            
        except Exception as e:
            logger.error(f"分析文件异常 {file_path}: {e}")
    
    def _extract_module_name(self, file_path: Path) -> str:
        """提取模块名"""
        parts = list(file_path.parts)
        
        # 移除文件扩展名
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        
        # 处理特殊情况
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        return '.'.join(parts)
    
    def _extract_imports(self, content: str) -> List[str]:
        """提取导入语句"""
        imports = []
        
        # 简单的正则表达式匹配
        import_lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                import_lines.append(line)
        
        # 提取模块名
        for line in import_lines:
            if line.startswith('from '):
                # from module import ...
                parts = line.split()
                if len(parts) >= 2:
                    module = parts[1]
                    imports.append(module)
            elif line.startswith('import '):
                # import module
                parts = line.split()
                if len(parts) >= 2:
                    module = parts[1].split('.')[0]
                    imports.append(module)
        
        return list(set(imports))  # 去重
    
    def _find_hardcoded_paths(self, content: str) -> List[str]:
        """查找硬编码路径"""
        hardcoded_paths = []
        
        # 查找 /workspace/code 路径
        import re
        pattern = r'/workspace/code[^\s\'"]*'
        matches = re.findall(pattern, content)
        hardcoded_paths.extend(matches)
        
        # 查找其他硬编码路径模式
        patterns = [
            r'["\'](/[^\'"]+)["\']',  # 绝对路径
            r'sys\.path\.append\(["\']([^"\']+)["\']',  # sys.path.append
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            hardcoded_paths.extend(matches)
        
        return list(set(hardcoded_paths))
    
    def _analyze_error_patterns(self, content: str) -> str:
        """分析错误处理模式"""
        if 'try:' in content and 'except ImportError' in content:
            return 'import_fallback'
        elif 'try:' in content and 'except:' in content:
            return 'generic_exception'
        elif 'except Exception' in content:
            return 'specific_exception'
        else:
            return 'no_error_handling'
    
    def build_dependency_graph(self):
        """构建依赖关系图"""
        self.dependency_graph.clear()
        
        # 添加节点
        for module_name in self.modules.keys():
            self.dependency_graph.add_node(module_name)
        
        # 添加边（依赖关系）
        for module_name, module_dep in self.modules.items():
            for dep in module_dep.dependencies:
                if dep in self.modules:
                    self.dependency_graph.add_edge(module_name, dep)
        
        # 计算被依赖关系
        for module_name, module_dep in self.modules.items():
            module_dep.dependents = list(self.dependency_graph.predecessors(module_name))
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """查找循环依赖"""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return cycles
        except Exception as e:
            logger.error(f"查找循环依赖失败: {e}")
            return []
    
    def calculate_coupling_metrics(self) -> Dict[str, float]:
        """计算耦合度指标"""
        metrics = {}
        
        # 总体耦合度
        total_possible_edges = len(self.modules) * (len(self.modules) - 1)
        actual_edges = self.dependency_graph.number_of_edges()
        metrics['overall_coupling'] = actual_edges / max(total_possible_edges, 1)
        
        # 平均入度（被依赖程度）
        in_degrees = [self.dependency_graph.in_degree(node) for node in self.dependency_graph.nodes()]
        metrics['avg_in_degree'] = sum(in_degrees) / max(len(in_degrees), 1)
        
        # 平均出度（依赖程度）
        out_degrees = [self.dependency_graph.out_degree(node) for node in self.dependency_graph.nodes()]
        metrics['avg_out_degree'] = sum(out_degrees) / max(len(out_degrees), 1)
        
        # 孤立节点数
        isolated_nodes = list(nx.isolates(self.dependency_graph))
        metrics['isolated_nodes'] = len(isolated_nodes)
        
        return metrics
    
    def generate_dependency_report(self) -> Dict[str, any]:
        """生成依赖分析报告"""
        self.build_dependency_graph()
        circular_deps = self.find_circular_dependencies()
        coupling_metrics = self.calculate_coupling_metrics()
        
        # 统计硬编码路径
        hardcoded_stats = {
            'total_files': len(self.modules),
            'files_with_hardcoded_paths': sum(1 for m in self.modules.values() if m.has_hardcoded_paths),
            'total_hardcoded_paths': len(self.hardcoded_paths),
            'hardcoded_path_files': [m.file_path for m in self.modules.values() if m.has_hardcoded_paths]
        }
        
        # 统计错误处理模式
        error_handling_stats = {}
        for pattern in ['import_fallback', 'generic_exception', 'specific_exception', 'no_error_handling']:
            count = sum(1 for m in self.modules.values() if m.error_handling_pattern == pattern)
            error_handling_stats[pattern] = count
        
        report = {
            'analysis_time': datetime.now().isoformat(),
            'total_modules': len(self.modules),
            'dependency_graph': {
                'nodes': list(self.dependency_graph.nodes()),
                'edges': list(self.dependency_graph.edges()),
                'circular_dependencies': circular_deps
            },
            'coupling_metrics': coupling_metrics,
            'hardcoded_paths': hardcoded_stats,
            'error_handling_patterns': error_handling_stats,
            'modules': {name: {
                'file_path': dep.file_path,
                'dependencies': dep.dependencies,
                'dependents': dep.dependents,
                'has_hardcoded_paths': dep.has_hardcoded_paths,
                'error_handling_pattern': dep.error_handling_pattern
            } for name, dep in self.modules.items()}
        }
        
        return report


class DependencyVisualizer:
    """依赖关系可视化器"""
    
    def __init__(self, analyzer: DependencyAnalyzer):
        self.analyzer = analyzer
    
    def plot_dependency_graph(self, output_path: str, title: str = "模块依赖关系图"):
        """绘制依赖关系图"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # 使用spring布局
        pos = nx.spring_layout(self.analyzer.dependency_graph, k=3, iterations=50)
        
        # 绘制节点
        node_colors = []
        node_sizes = []
        
        for node in self.analyzer.dependency_graph.nodes():
            module = self.analyzer.modules.get(node)
            if module:
                if module.has_hardcoded_paths:
                    node_colors.append('red')
                else:
                    node_colors.append('lightblue')
                
                # 根据依赖数量调整节点大小
                in_degree = self.analyzer.dependency_graph.in_degree(node)
                out_degree = self.analyzer.dependency_graph.out_degree(node)
                node_sizes.append(300 + in_degree * 50 + out_degree * 30)
            else:
                node_colors.append('gray')
                node_sizes.append(300)
        
        nx.draw_networkx_nodes(
            self.analyzer.dependency_graph, 
            pos, 
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            self.analyzer.dependency_graph,
            pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            alpha=0.6,
            ax=ax
        )
        
        # 绘制标签
        labels = {}
        for node in self.analyzer.dependency_graph.nodes():
            # 简化标签
            labels[node] = node.split('.')[-1] if '.' in node else node
        
        nx.draw_networkx_labels(
            self.analyzer.dependency_graph,
            pos,
            labels,
            font_size=8,
            ax=ax
        )
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color='lightblue', label='正常模块'),
            mpatches.Patch(color='red', label='含硬编码路径'),
            mpatches.Patch(color='gray', label='外部依赖')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"依赖关系图已保存: {output_path}")
    
    def plot_coupling_metrics(self, output_path: str):
        """绘制耦合度指标图"""
        metrics = self.analyzer.calculate_coupling_metrics()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 总体耦合度
        ax1.bar(['总体耦合度'], [metrics['overall_coupling']], color='skyblue')
        ax1.set_ylabel('耦合度')
        ax1.set_title('模块间总体耦合度')
        ax1.set_ylim(0, 1)
        
        # 2. 平均入度和出度
        ax2.bar(['平均入度', '平均出度'], 
               [metrics['avg_in_degree'], metrics['avg_out_degree']], 
               color=['lightcoral', 'lightgreen'])
        ax2.set_ylabel('度数')
        ax2.set_title('平均依赖关系度数')
        
        # 3. 节点类型分布
        node_types = ['正常模块', '含硬编码路径', '孤立节点']
        counts = [
            len(self.analyzer.modules) - metrics['isolated_nodes'] - sum(1 for m in self.analyzer.modules.values() if m.has_hardcoded_paths),
            sum(1 for m in self.analyzer.modules.values() if m.has_hardcoded_paths),
            metrics['isolated_nodes']
        ]
        ax3.pie(counts, labels=node_types, autopct='%1.1f%%', startangle=90)
        ax3.set_title('模块类型分布')
        
        # 4. 错误处理模式分布
        pattern_counts = []
        pattern_labels = []
        pattern_distribution = {}
        for module_name, pattern in self.analyzer.error_patterns.items():
            pattern_distribution[pattern] = pattern_distribution.get(pattern, 0) + 1
        
        for pattern, count in pattern_distribution.items():
            pattern_counts.append(count)
            pattern_labels.append(pattern)
        
        if pattern_counts:
            ax4.bar(pattern_labels, pattern_counts, color='orange', alpha=0.7)
            ax4.set_ylabel('模块数量')
            ax4.set_title('错误处理模式分布')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, '无错误处理模式数据', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes)
            ax4.set_title('错误处理模式分布')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"耦合度指标图已保存: {output_path}")
    
    def plot_hardcoded_paths_analysis(self, output_path: str):
        """绘制硬编码路径分析图"""
        # 统计各模块的硬编码路径数量
        module_path_counts = {}
        for module_name, module in self.analyzer.modules.items():
            if module.has_hardcoded_paths:
                # 计算该模块的硬编码路径数量
                path_count = sum(1 for path in self.analyzer.hardcoded_paths 
                               if module.file_path in path or module_name in path)
                module_path_counts[module_name] = path_count
        
        if not module_path_counts:
            logger.warning("未发现硬编码路径")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 各模块硬编码路径数量
        modules = list(module_path_counts.keys())[:20]  # 只显示前20个
        counts = [module_path_counts[m] for m in modules]
        
        ax1.barh(modules, counts, color='red', alpha=0.7)
        ax1.set_xlabel('硬编码路径数量')
        ax1.set_title('各模块硬编码路径数量（前20个）')
        
        # 2. 硬编码路径类型分布
        path_types = {}
        for path in self.analyzer.hardcoded_paths:
            if '/workspace/code' in path:
                path_types['项目路径'] = path_types.get('项目路径', 0) + 1
            elif path.startswith('/'):
                path_types['绝对路径'] = path_types.get('绝对路径', 0) + 1
            else:
                path_types['其他路径'] = path_types.get('其他路径', 0) + 1
        
        if path_types:
            ax2.pie(path_types.values(), labels=path_types.keys(), autopct='%1.1f%%')
            ax2.set_title('硬编码路径类型分布')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"硬编码路径分析图已保存: {output_path}")


def main():
    """主函数 - 生成依赖关系分析报告"""
    print("=" * 60)
    print("智能交通流预测系统依赖关系分析")
    print("=" * 60)
    
    # 创建分析器
    analyzer = DependencyAnalyzer()
    
    # 分析代码库
    code_root = Path("/workspace/code")
    if not code_root.exists():
        print(f"代码库路径不存在: {code_root}")
        return
    
    print(f"\n开始分析代码库: {code_root}")
    modules = analyzer.analyze_codebase(code_root)
    
    # 生成报告
    print("\n生成依赖分析报告...")
    report = analyzer.generate_dependency_report()
    
    # 保存报告
    report_path = Path("/workspace/code/refactored_architecture/dependency_analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"依赖分析报告已保存: {report_path}")
    
    # 创建可视化器
    visualizer = DependencyVisualizer(analyzer)
    
    # 生成图表
    print("\n生成可视化图表...")
    
    # 1. 依赖关系图
    graph_path = "/workspace/code/refactored_architecture/dependency_graph.png"
    visualizer.plot_dependency_graph(graph_path, "重构前模块依赖关系图")
    
    # 2. 耦合度指标图
    metrics_path = "/workspace/code/refactored_architecture/coupling_metrics.png"
    visualizer.plot_coupling_metrics(metrics_path)
    
    # 3. 硬编码路径分析图
    hardcoded_path = "/workspace/code/refactored_architecture/hardcoded_paths_analysis.png"
    visualizer.plot_hardcoded_paths_analysis(hardcoded_path)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("分析结果统计")
    print("=" * 60)
    print(f"总模块数: {report['total_modules']}")
    print(f"依赖关系数: {report['dependency_graph']['edges']}")
    print(f"循环依赖数: {len(report['dependency_graph']['circular_dependencies'])}")
    print(f"含硬编码路径的模块: {report['hardcoded_paths']['files_with_hardcoded_paths']}")
    print(f"硬编码路径总数: {report['hardcoded_paths']['total_hardcoded_paths']}")
    print(f"总体耦合度: {report['coupling_metrics']['overall_coupling']:.3f}")
    
    # 显示循环依赖
    if report['dependency_graph']['circular_dependencies']:
        print("\n发现循环依赖:")
        for i, cycle in enumerate(report['dependency_graph']['circular_dependencies'], 1):
            print(f"  {i}. {' -> '.join(cycle)} -> {cycle[0]}")
    
    # 显示错误处理模式分布
    print("\n错误处理模式分布:")
    for pattern, count in report['error_handling_patterns'].items():
        print(f"  {pattern}: {count} 个模块")
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()