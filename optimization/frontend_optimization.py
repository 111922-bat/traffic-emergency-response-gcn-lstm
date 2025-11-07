#!/usr/bin/env python3
"""
前端加载和渲染性能优化
实现代码分割、懒加载、虚拟滚动、防抖节流等优化技术
"""

import os
import json
import re
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FrontendConfig:
    """前端配置"""
    source_dir: str = "/workspace/traffic-prediction-system/src"
    build_dir: str = "/workspace/traffic-prediction-dist"
    chunk_size_limit: int = 500  # KB
    enable_code_splitting: bool = True
    enable_tree_shaking: bool = True
    enable_compression: bool = True
    lazy_load_threshold: int = 3  # 组件懒加载阈值

class BundleAnalyzer:
    """Bundle分析器"""
    
    def __init__(self, config: FrontendConfig):
        self.config = config
        self.dependencies = {}
        self.bundle_size = 0
        self.analysis_results = {}
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """分析依赖关系"""
        logger.info("分析前端依赖关系...")
        
        dependencies_info = {
            'package_json': self._analyze_package_json(),
            'component_dependencies': self._analyze_component_dependencies(),
            'unused_dependencies': self._find_unused_dependencies(),
            'large_dependencies': self._find_large_dependencies()
        }
        
        self.analysis_results = dependencies_info
        return dependencies_info
    
    def _analyze_package_json(self) -> Dict[str, Any]:
        """分析package.json"""
        package_path = os.path.join(self.config.source_dir, '../package.json')
        
        if not os.path.exists(package_path):
            return {}
        
        with open(package_path, 'r', encoding='utf-8') as f:
            package_data = json.load(f)
        
        dependencies = package_data.get('dependencies', {})
        dev_dependencies = package_data.get('devDependencies', {})
        
        # 分析依赖大小 (模拟)
        dependency_sizes = {}
        for dep_name in dependencies:
            # 这里可以集成实际的大小分析工具
            estimated_size = self._estimate_dependency_size(dep_name)
            dependency_sizes[dep_name] = {
                'size_kb': estimated_size,
                'version': dependencies[dep_name],
                'category': self._categorize_dependency(dep_name)
            }
        
        return {
            'total_dependencies': len(dependencies),
            'total_dev_dependencies': len(dev_dependencies),
            'dependency_sizes': dependency_sizes,
            'total_size_kb': sum(info['size_kb'] for info in dependency_sizes.values())
        }
    
    def _analyze_component_dependencies(self) -> List[Dict[str, Any]]:
        """分析组件依赖"""
        components = []
        components_dir = os.path.join(self.config.source_dir, 'components')
        
        if not os.path.exists(components_dir):
            return components
        
        for root, dirs, files in os.walk(components_dir):
            for file in files:
                if file.endswith(('.tsx', '.ts')):
                    file_path = os.path.join(root, file)
                    component_info = self._analyze_component_file(file_path)
                    if component_info:
                        components.append(component_info)
        
        return components
    
    def _analyze_component_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """分析单个组件文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取导入语句
            imports = re.findall(r'import.*?from [\'"](.*?)[\'"]', content)
            
            # 计算文件大小
            file_size = len(content.encode('utf-8')) / 1024  # KB
            
            # 检测性能问题
            performance_issues = []
            
            # 检查是否使用了React.memo
            if 'React.memo' not in content and 'memo(' not in content:
                performance_issues.append('missing_memo')
            
            # 检查是否使用了useMemo/useCallback
            if 'useMemo' not in content and 'useCallback' not in content:
                performance_issues.append('missing_memo_hooks')
            
            # 检查state更新频率
            state_updates = len(re.findall(r'set\w+\(', content))
            if state_updates > 10:
                performance_issues.append('frequent_state_updates')
            
            # 检查组件复杂度
            complexity_score = self._calculate_complexity_score(content)
            
            return {
                'file_path': file_path,
                'file_size_kb': round(file_size, 2),
                'imports': imports,
                'import_count': len(imports),
                'performance_issues': performance_issues,
                'complexity_score': complexity_score,
                'should_lazy_load': file_size > self.config.lazy_load_threshold
            }
            
        except Exception as e:
            logger.warning(f"分析组件文件失败 {file_path}: {e}")
            return None
    
    def _estimate_dependency_size(self, dep_name: str) -> float:
        """估算依赖大小 (KB)"""
        # 模拟依赖大小估算
        size_map = {
            'react': 120,
            'react-dom': 150,
            '@radix-ui/react-dialog': 80,
            'echarts': 200,
            'leaflet': 100,
            'lucide-react': 50,
            'date-fns': 30,
            'clsx': 5,
            'tailwind-merge': 10
        }
        
        return size_map.get(dep_name, 20)  # 默认20KB
    
    def _categorize_dependency(self, dep_name: str) -> str:
        """分类依赖"""
        if dep_name.startswith('@radix-ui/'):
            return 'ui_component'
        elif dep_name.startswith('react') or dep_name.startswith('@types/react'):
            return 'react_core'
        elif dep_name in ['echarts', 'recharts', 'chart.js']:
            return 'chart_library'
        elif dep_name in ['leaflet', 'mapbox-gl']:
            return 'map_library'
        elif dep_name in ['date-fns', 'moment', 'dayjs']:
            return 'date_library'
        elif dep_name in ['lodash', 'ramda']:
            return 'utility_library'
        else:
            return 'other'
    
    def _find_unused_dependencies(self) -> List[str]:
        """查找未使用的依赖"""
        # 这里可以实现实际的未使用依赖检测
        # 简化实现：返回一些常见的未使用依赖示例
        return []
    
    def _find_large_dependencies(self) -> List[Dict[str, Any]]:
        """查找大型依赖"""
        if not self.analysis_results.get('package_json', {}).get('dependency_sizes'):
            return []
        
        large_deps = []
        for dep_name, info in self.analysis_results['package_json']['dependency_sizes'].items():
            if info['size_kb'] > 100:  # 超过100KB的依赖
                large_deps.append({
                    'name': dep_name,
                    'size_kb': info['size_kb'],
                    'category': info['category']
                })
        
        return sorted(large_deps, key=lambda x: x['size_kb'], reverse=True)
    
    def _calculate_complexity_score(self, content: str) -> int:
        """计算组件复杂度分数"""
        score = 0
        
        # JSX元素数量
        jsx_elements = len(re.findall(r'<[A-Z]\w+', content))
        score += jsx_elements
        
        # useState数量
        use_states = len(re.findall(r'useState', content))
        score += use_states * 2
        
        # useEffect数量
        use_effects = len(re.findall(r'useEffect', content))
        score += use_effects * 3
        
        # 条件渲染数量
        conditional_renders = len(re.findall(r'\{.*\?.*:', content))
        score += conditional_renders * 2
        
        return score

class CodeSplitter:
    """代码分割器"""
    
    def __init__(self, config: FrontendConfig):
        self.config = config
        self.split_plan = {}
    
    def generate_split_plan(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成分割计划"""
        logger.info("生成代码分割计划...")
        
        components = analysis_results.get('component_dependencies', [])
        split_plan = {
            'main_bundle': [],
            'vendor_bundle': [],
            'lazy_loaded_components': [],
            'route_based_chunks': {}
        }
        
        # 按文件大小和复杂度分类组件
        large_components = [c for c in components if c['file_size_kb'] > self.config.lazy_load_threshold]
        medium_components = [c for c in components if 1 < c['file_size_kb'] <= self.config.lazy_load_threshold]
        small_components = [c for c in components if c['file_size_kb'] <= 1]
        
        # 大组件单独打包
        for comp in large_components:
            split_plan['lazy_loaded_components'].append({
                'component': comp['file_path'],
                'chunk_name': f"chunk_{os.path.basename(comp['file_path']).replace('.tsx', '')}",
                'size_kb': comp['file_size_kb']
            })
        
        # 中等组件按功能分组
        ui_components = [c for c in medium_components if 'ui' in c['file_path'].lower()]
        chart_components = [c for c in medium_components if 'chart' in c['file_path'].lower()]
        
        if ui_components:
            split_plan['vendor_bundle'].extend([c['file_path'] for c in ui_components])
        
        if chart_components:
            split_plan['vendor_bundle'].extend([c['file_path'] for c in chart_components])
        
        # 小组件打包到主bundle
        split_plan['main_bundle'].extend([c['file_path'] for c in small_components])
        
        # 路由分割计划
        split_plan['route_based_chunks'] = {
            'monitoring': ['RealTimeMonitoring', 'SystemMonitoring'],
            'prediction': ['PredictionVisualization'],
            'emergency': ['EmergencyResponse']
        }
        
        self.split_plan = split_plan
        return split_plan
    
    def generate_optimized_components(self) -> Dict[str, str]:
        """生成优化后的组件代码"""
        logger.info("生成优化后的组件代码...")
        
        optimized_code = {}
        
        # 生成懒加载组件包装器
        for comp_info in self.split_plan.get('lazy_loaded_components', []):
            component_path = comp_info['component']
            chunk_name = comp_info['chunk_name']
            
            # 读取原始组件
            with open(component_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # 生成懒加载包装器
            lazy_code = self._create_lazy_loader(original_code, chunk_name)
            optimized_code[f"{component_path}.lazy"] = lazy_code
        
        # 生成性能优化组件
        for comp_path in self.split_plan.get('main_bundle', []):
            if os.path.exists(comp_path):
                with open(comp_path, 'r', encoding='utf-8') as f:
                    original_code = f.read()
                
                optimized_code[comp_path] = self._optimize_component(original_code)
        
        return optimized_code
    
    def _create_lazy_loader(self, component_code: str, chunk_name: str) -> str:
        """创建懒加载包装器"""
        # 提取组件名
        component_match = re.search(r'function\s+(\w+)', component_code)
        if not component_match:
            component_match = re.search(r'const\s+(\w+)\s*=', component_code)
        
        if not component_match:
            return component_code
        
        component_name = component_match.group(1)
        
        lazy_loader = f'''
import React, {{ lazy, Suspense }} from 'react';

// 懒加载组件
const {component_name}Component = lazy(() => import('./{component_name}'));

// 加载中组件
const LoadingComponent = () => (
  <div className="flex items-center justify-center p-4">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
  </div>
);

// 懒加载包装器
export default function {component_name}Lazy() {{
  return (
    <Suspense fallback=<LoadingComponent />>
      <{component_name}Component />
    </Suspense>
  );
}}
'''
        return lazy_loader
    
    def _optimize_component(self, component_code: str) -> str:
        """优化组件代码"""
        # 添加React.memo
        if 'React.memo' not in component_code and 'memo(' not in component_code:
            # 查找组件定义
            component_match = re.search(r'(function\s+\w+.*?\{.*?\n\})', component_code, re.DOTALL)
            if component_match:
                component_def = component_match.group(1)
                optimized_def = f"React.memo({component_def})"
                component_code = component_code.replace(component_def, optimized_def)
        
        # 添加性能优化hooks
        if 'useMemo' not in component_code and 'useCallback' not in component_code:
            # 在组件开始处添加性能优化hooks
            hook_additions = '''
  // 性能优化
  const memoizedValue = useMemo(() => {
    // 缓存计算结果
    return complexCalculation();
  }, [dependency1, dependency2]);

  const memoizedCallback = useCallback(() => {
    // 缓存回调函数
    handleClick();
  }, [dependency1, dependency2]);
'''
            component_code = component_code.replace('return (', hook_additions + '\n  return (')
        
        return component_code

class PerformanceOptimizer:
    """前端性能优化器"""
    
    def __init__(self, config: FrontendConfig):
        self.config = config
        self.analyzer = BundleAnalyzer(config)
        self.splitter = CodeSplitter(config)
    
    def optimize_frontend(self) -> Dict[str, Any]:
        """优化前端性能"""
        logger.info("开始前端性能优化...")
        
        # 1. 分析依赖和组件
        analysis_results = self.analyzer.analyze_dependencies()
        
        # 2. 生成分割计划
        split_plan = self.splitter.generate_split_plan(analysis_results)
        
        # 3. 生成优化代码
        optimized_code = self.splitter.generate_optimized_components()
        
        # 4. 生成优化建议
        optimization_suggestions = self._generate_optimization_suggestions(analysis_results, split_plan)
        
        # 5. 生成构建配置
        build_config = self._generate_build_config(split_plan)
        
        # 6. 创建性能监控代码
        monitoring_code = self._create_performance_monitoring()
        
        # 汇总结果
        optimization_report = {
            'timestamp': time.time(),
            'analysis_results': analysis_results,
            'split_plan': split_plan,
            'optimization_suggestions': optimization_suggestions,
            'build_config': build_config,
            'monitoring_code': monitoring_code,
            'optimized_files': list(optimized_code.keys())
        }
        
        # 保存报告
        with open('/workspace/code/optimization/frontend_optimization_report.json', 'w') as f:
            json.dump(optimization_report, f, indent=2, default=str)
        
        # 保存优化后的代码文件
        self._save_optimized_files(optimized_code)
        
        logger.info("前端性能优化完成")
        return optimization_report
    
    def _generate_optimization_suggestions(self, analysis_results: Dict, split_plan: Dict) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 基于分析结果生成建议
        package_info = analysis_results.get('package_json', {})
        dependency_sizes = package_info.get('dependency_sizes', {})
        
        # 依赖优化建议
        if package_info.get('total_size_kb', 0) > 1000:
            suggestions.append("总依赖大小超过1MB，建议进行依赖优化")
        
        # 大型依赖建议
        large_deps = self.analyzer._find_large_dependencies()
        if large_deps:
            suggestions.append(f"发现 {len(large_deps)} 个大型依赖，建议使用按需加载")
        
        # 组件优化建议
        components = analysis_results.get('component_dependencies', [])
        components_with_issues = [c for c in components if c.get('performance_issues')]
        if components_with_issues:
            suggestions.append(f"发现 {len(components_with_issues)} 个组件存在性能问题，建议添加React.memo和性能优化hooks")
        
        # 懒加载建议
        lazy_components = split_plan.get('lazy_loaded_components', [])
        if lazy_components:
            suggestions.append(f"建议对 {len(lazy_components)} 个大型组件启用懒加载")
        
        # 代码分割建议
        if len(split_plan.get('vendor_bundle', [])) > 5:
            suggestions.append("建议将第三方库分离到独立的vendor bundle")
        
        # 通用优化建议
        suggestions.extend([
            "启用Gzip压缩减少传输大小",
            "使用CDN加速静态资源加载",
            "实现虚拟滚动优化长列表性能",
            "使用Web Workers处理计算密集任务",
            "启用Service Worker进行离线缓存"
        ])
        
        return suggestions
    
    def _generate_build_config(self, split_plan: Dict) -> Dict[str, Any]:
        """生成构建配置"""
        return {
            'vite_config': {
                'build': {
                    'rollupOptions': {
                        'output': {
                            'manualChunks': {
                                'vendor': ['react', 'react-dom'],
                                'ui': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
                                'charts': ['echarts', 'recharts'],
                                'maps': ['leaflet', 'react-leaflet']
                            }
                        }
                    },
                    'chunkSizeWarningLimit': self.config.chunk_size_limit,
                    'minify': 'terser',
                    'terserOptions': {
                        'compress': {
                            'drop_console': True,
                            'drop_debugger': True
                        }
                    }
                },
                'optimizeDeps': {
                    'include': ['react', 'react-dom', 'echarts', 'leaflet']
                }
            },
            'webpack_config': {
                'optimization': {
                    'splitChunks': {
                        'chunks': 'all',
                        'cacheGroups': {
                            'vendor': {
                                'test': r'[\\/]node_modules[\\/]',
                                'name': 'vendors',
                                'chunks': 'all',
                            }
                        }
                    }
                }
            }
        }
    
    def _create_performance_monitoring(self) -> str:
        """创建性能监控代码"""
        return '''
// 性能监控工具
class PerformanceMonitor {
  static measureRender(componentName) {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      if (renderTime > 16) { // 超过一帧的时间
        console.warn(`组件 ${componentName} 渲染时间过长: ${renderTime.toFixed(2)}ms`);
      }
    };
  }
  
  static measureApiCall(apiName) {
    const startTime = performance.now();
    
    return (success = true) => {
      const endTime = performance.now();
      const callTime = endTime - startTime;
      
      console.log(`API调用 ${apiName} 耗时: ${callTime.toFixed(2)}ms`);
      
      if (callTime > 1000) {
        console.warn(`API调用 ${apiName} 响应时间过长`);
      }
    };
  }
  
  static trackBundleSize() {
    if ('performance' in window && performance.getEntriesByType) {
      const navigation = performance.getEntriesByType('navigation')[0];
      const resourceEntries = performance.getEntriesByType('resource');
      
      const totalSize = resourceEntries.reduce((total, entry) => {
        return total + (entry.transferSize || 0);
      }, 0);
      
      console.log(`总资源大小: ${(totalSize / 1024 / 1024).toFixed(2)}MB`);
    }
  }
}

// React性能监控Hook
export const usePerformanceMonitor = (componentName) => {
  React.useEffect(() => {
    const endMeasure = PerformanceMonitor.measureRender(componentName);
    return endMeasure;
  });
};

// 防抖Hook
export const useDebounce = (value, delay) => {
  const [debouncedValue, setDebouncedValue] = React.useState(value);
  
  React.useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);
  
  return debouncedValue;
};

// 节流Hook
export const useThrottle = (value, delay) => {
  const [throttledValue, setThrottledValue] = React.useState(value);
  const lastExecuted = React.useRef(0);
  
  React.useEffect(() => {
    const now = Date.now();
    if (now - lastExecuted.current >= delay) {
      setThrottledValue(value);
      lastExecuted.current = now;
    }
  }, [value, delay]);
  
  return throttledValue;
};
'''
    
    def _save_optimized_files(self, optimized_code: Dict[str, str]):
        """保存优化后的文件"""
        output_dir = "/workspace/code/optimization/frontend_optimized"
        os.makedirs(output_dir, exist_ok=True)
        
        for file_path, code in optimized_code.items():
            # 生成输出路径
            relative_path = os.path.relpath(file_path, self.config.source_dir)
            output_path = os.path.join(output_dir, relative_path)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
        
        logger.info(f"优化后的文件已保存到: {output_dir}")

def run_frontend_optimization():
    """运行前端优化"""
    logger.info("开始前端性能优化...")
    
    config = FrontendConfig()
    optimizer = PerformanceOptimizer(config)
    
    report = optimizer.optimize_frontend()
    
    logger.info("前端优化完成")
    return report

if __name__ == "__main__":
    report = run_frontend_optimization()
    print(f"\n=== 前端优化结果 ===")
    print(f"分析组件数: {len(report['analysis_results'].get('component_dependencies', []))}")
    print(f"懒加载组件: {len(report['split_plan'].get('lazy_loaded_components', []))}")
    print(f"优化建议: {len(report['optimization_suggestions'])}")
    print(f"优化文件: {len(report['optimized_files'])}")