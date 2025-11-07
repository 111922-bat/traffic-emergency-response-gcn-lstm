#!/usr/bin/env python3
"""
错误处理和优雅降级系统
Error Handling and Graceful Degradation System

提供统一的错误处理、优雅降级、故障恢复机制
"""

import asyncio
import logging
import traceback
import time
import functools
from typing import Dict, List, Optional, Any, Callable, Type, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json
import threading
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别"""
    NETWORK = "network"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    VALIDATION = "validation"
    SYSTEM = "system"
    BUSINESS_LOGIC = "business_logic"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"


class DegradationLevel(Enum):
    """降级级别"""
    NONE = "none"           # 无降级
    PARTIAL = "partial"     # 部分降级
    MINIMAL = "minimal"     # 最小化服务
    EMERGENCY = "emergency" # 紧急模式


@dataclass
class ErrorInfo:
    """错误信息"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.stack_trace:
            self.stack_trace = traceback.format_exc()


class ErrorHandler(ABC):
    """错误处理器抽象类"""
    
    @abstractmethod
    async def handle_error(self, error: Exception, context: str) -> Tuple[bool, Any]:
        """处理错误，返回是否已处理及处理结果"""
        pass


class RetryErrorHandler(ErrorHandler):
    """重试错误处理器"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0, 
                 backoff_factor: float = 2.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retry_counts = defaultdict(int)
    
    async def handle_error(self, error: Exception, context: str) -> Tuple[bool, Any]:
        """重试处理错误"""
        retry_count = self.retry_counts[context]
        
        if retry_count >= self.max_retries:
            logger.error(f"重试次数已达上限 {self.max_retries}，停止重试: {context}")
            return False, None
        
        # 计算延迟时间
        delay = min(self.delay * (self.backoff_factor ** retry_count), self.max_delay)
        
        logger.warning(f"错误处理重试 {retry_count + 1}/{self.max_retries}，延迟 {delay:.2f}s: {context} - {error}")
        
        await asyncio.sleep(delay)
        self.retry_counts[context] += 1
        
        return True, None  # 表示需要重试


class FallbackErrorHandler(ErrorHandler):
    """回退错误处理器"""
    
    def __init__(self, fallback_values: Dict[str, Any]):
        self.fallback_values = fallback_values
    
    async def handle_error(self, error: Exception, context: str) -> Tuple[bool, Any]:
        """使用回退值处理错误"""
        if context in self.fallback_values:
            logger.warning(f"使用回退值处理错误: {context}")
            return True, self.fallback_values[context]
        
        logger.error(f"未找到回退值: {context}")
        return False, None


class CircuitBreakerErrorHandler(ErrorHandler):
    """熔断器错误处理器"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts = defaultdict(int)
        self.last_failure_time = defaultdict(float)
        self.circuit_states = defaultdict(lambda: "closed")
    
    async def handle_error(self, error: Exception, context: str) -> Tuple[bool, Any]:
        """熔断器处理错误"""
        current_time = time.time()
        
        # 检查是否在恢复期内
        if (self.circuit_states[context] == "open" and 
            current_time - self.last_failure_time[context] < self.recovery_timeout):
            logger.warning(f"熔断器开启，跳过请求: {context}")
            return False, None
        
        # 检查失败次数
        if self.failure_counts[context] >= self.failure_threshold:
            self.circuit_states[context] = "open"
            self.last_failure_time[context] = current_time
            logger.error(f"熔断器开启: {context}")
            return False, None
        
        # 记录失败
        self.failure_counts[context] += 1
        self.last_failure_time[context] = current_time
        
        # 如果失败次数达到阈值，开启熔断器
        if self.failure_counts[context] >= self.failure_threshold:
            self.circuit_states[context] = "open"
            logger.error(f"熔断器开启: {context}")
        
        return False, None
    
    def reset_circuit(self, context: str):
        """重置熔断器"""
        self.failure_counts[context] = 0
        self.circuit_states[context] = "closed"
        logger.info(f"重置熔断器: {context}")


class DegradationManager:
    """优雅降级管理器"""
    
    def __init__(self):
        self.degradation_levels = {
            DegradationLevel.NONE: self._handle_none,
            DegradationLevel.PARTIAL: self._handle_partial,
            DegradationLevel.MINIMAL: self._handle_minimal,
            DegradationLevel.EMERGENCY: self._handle_emergency
        }
        self.current_level = DegradationLevel.NONE
        self.service_states = defaultdict(lambda: {"status": "healthy", "degraded": False})
        self.degradation_rules = {}
        self.logger = logging.getLogger(__name__)
    
    def set_degradation_level(self, level: DegradationLevel):
        """设置降级级别"""
        if level != self.current_level:
            old_level = self.current_level
            self.current_level = level
            self.logger.warning(f"降级级别变更: {old_level.value} -> {level.value}")
            self._notify_degradation_change(old_level, level)
    
    def get_current_level(self) -> DegradationLevel:
        """获取当前降级级别"""
        return self.current_level
    
    def register_service(self, service_name: str, degradation_handler: Callable):
        """注册服务降级处理器"""
        self.degradation_rules[service_name] = degradation_handler
        self.logger.info(f"注册服务降级处理器: {service_name}")
    
    def mark_service_unhealthy(self, service_name: str):
        """标记服务不健康"""
        self.service_states[service_name]["status"] = "unhealthy"
        self.service_states[service_name]["degraded"] = True
        self._evaluate_degradation_level()
        self.logger.warning(f"服务标记为不健康: {service_name}")
    
    def mark_service_healthy(self, service_name: str):
        """标记服务健康"""
        self.service_states[service_name]["status"] = "healthy"
        self.service_states[service_name]["degraded"] = False
        self._evaluate_degradation_level()
        self.logger.info(f"服务标记为健康: {service_name}")
    
    def _evaluate_degradation_level(self):
        """评估降级级别"""
        unhealthy_count = sum(1 for state in self.service_states.values() 
                            if state["status"] == "unhealthy")
        total_services = len(self.service_states)
        
        if total_services == 0:
            new_level = DegradationLevel.NONE
        elif unhealthy_count == 0:
            new_level = DegradationLevel.NONE
        elif unhealthy_count / total_services <= 0.3:
            new_level = DegradationLevel.PARTIAL
        elif unhealthy_count / total_services <= 0.7:
            new_level = DegradationLevel.MINIMAL
        else:
            new_level = DegradationLevel.EMERGENCY
        
        if new_level != self.current_level:
            self.set_degradation_level(new_level)
    
    def _notify_degradation_change(self, old_level: DegradationLevel, new_level: DegradationLevel):
        """通知降级级别变化"""
        for service_name, handler in self.degradation_rules.items():
            try:
                handler(old_level, new_level)
            except Exception as e:
                self.logger.error(f"降级处理器执行失败 {service_name}: {e}")
    
    def _handle_none(self, service_name: str):
        """处理无降级"""
        pass
    
    def _handle_partial(self, service_name: str):
        """处理部分降级"""
        self.logger.info(f"部分降级服务: {service_name}")
    
    def _handle_minimal(self, service_name: str):
        """处理最小化服务"""
        self.logger.warning(f"最小化服务: {service_name}")
    
    def _handle_emergency(self, service_name: str):
        """处理紧急模式"""
        self.logger.error(f"紧急模式服务: {service_name}")


class ErrorRecoveryStrategy(ABC):
    """错误恢复策略"""
    
    @abstractmethod
    async def recover(self, error: Exception, context: str) -> bool:
        """执行恢复，返回是否成功"""
        pass


class HealthCheckRecoveryStrategy(ErrorRecoveryStrategy):
    """健康检查恢复策略"""
    
    def __init__(self, health_check_func: Callable):
        self.health_check_func = health_check_func
    
    async def recover(self, error: Exception, context: str) -> bool:
        """通过健康检查恢复"""
        try:
            is_healthy = await self.health_check_func()
            if is_healthy:
                logger.info(f"健康检查通过，恢复服务: {context}")
                return True
            else:
                logger.warning(f"健康检查失败: {context}")
                return False
        except Exception as e:
            logger.error(f"健康检查异常: {e}")
            return False


class RestartRecoveryStrategy(ErrorRecoveryStrategy):
    """重启恢复策略"""
    
    def __init__(self, restart_func: Callable):
        self.restart_func = restart_func
    
    async def recover(self, error: Exception, context: str) -> bool:
        """通过重启恢复"""
        try:
            await self.restart_func()
            logger.info(f"服务重启成功: {context}")
            return True
        except Exception as e:
            logger.error(f"服务重启失败: {e}")
            return False


class ErrorMonitor:
    """错误监控器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.error_history = deque(maxlen=window_size)
        self.error_counts = defaultdict(int)
        self.error_patterns = {}
        self.lock = threading.Lock()
    
    def record_error(self, error: Exception, context: str, category: ErrorCategory, 
                    severity: ErrorSeverity):
        """记录错误"""
        with self.lock:
            error_info = ErrorInfo(
                error_id=f"{context}_{int(time.time())}_{hash(str(error))}",
                category=category,
                severity=severity,
                message=str(error),
                details={"context": context, "type": type(error).__name__},
                timestamp=datetime.now()
            )
            
            self.error_history.append(error_info)
            self.error_counts[f"{category.value}_{context}"] += 1
            
            # 检测错误模式
            self._detect_error_patterns()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        with self.lock:
            now = datetime.now()
            recent_errors = [e for e in self.error_history 
                           if now - e.timestamp < timedelta(hours=1)]
            
            return {
                "total_errors": len(self.error_history),
                "recent_errors": len(recent_errors),
                "error_counts": dict(self.error_counts),
                "error_patterns": self.error_patterns,
                "timestamp": now.isoformat()
            }
    
    def _detect_error_patterns(self):
        """检测错误模式"""
        # 简单的模式检测：连续相同类型的错误
        if len(self.error_history) < 5:
            return
        
        recent_errors = list(self.error_history)[-10:]  # 最近10个错误
        
        # 检测连续错误
        consecutive_errors = 1
        for i in range(1, len(recent_errors)):
            if (recent_errors[i].category == recent_errors[i-1].category and
                recent_errors[i].severity == recent_errors[i-1].severity):
                consecutive_errors += 1
            else:
                consecutive_errors = 1
            
            if consecutive_errors >= 5:  # 连续5个相同错误
                pattern_key = f"{recent_errors[i].category.value}_{recent_errors[i].severity.value}"
                self.error_patterns[pattern_key] = {
                    "count": consecutive_errors,
                    "first_occurrence": recent_errors[i-consecutive_errors+1].timestamp,
                    "last_occurrence": recent_errors[i].timestamp
                }
                break


class GlobalErrorHandler:
    """全局错误处理器"""
    
    def __init__(self):
        self.error_handlers: List[ErrorHandler] = []
        self.recovery_strategies: List[ErrorRecoveryStrategy] = []
        self.degradation_manager = DegradationManager()
        self.error_monitor = ErrorMonitor()
        self.logger = logging.getLogger(__name__)
        
        # 添加默认处理器
        self.add_error_handler(RetryErrorHandler(max_retries=3))
        self.add_error_handler(CircuitBreakerErrorHandler(failure_threshold=5))
    
    def add_error_handler(self, handler: ErrorHandler):
        """添加错误处理器"""
        self.error_handlers.append(handler)
        self.logger.info(f"添加错误处理器: {handler.__class__.__name__}")
    
    def add_recovery_strategy(self, strategy: ErrorRecoveryStrategy):
        """添加恢复策略"""
        self.recovery_strategies.append(strategy)
        self.logger.info(f"添加恢复策略: {strategy.__class__.__name__}")
    
    async def handle_error(self, error: Exception, context: str, 
                          category: ErrorCategory = ErrorCategory.SYSTEM,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Tuple[bool, Any]:
        """处理错误"""
        # 记录错误
        self.error_monitor.record_error(error, context, category, severity)
        
        # 尝试错误处理器
        for handler in self.error_handlers:
            try:
                handled, result = await handler.handle_error(error, context)
                if handled:
                    self.logger.info(f"错误已处理 {handler.__class__.__name__}: {context}")
                    return True, result
            except Exception as handler_error:
                self.logger.error(f"错误处理器异常 {handler.__class__.__name__}: {handler_error}")
        
        # 尝试恢复策略
        for strategy in self.recovery_strategies:
            try:
                recovered = await strategy.recover(error, context)
                if recovered:
                    self.logger.info(f"错误已恢复 {strategy.__class__.__name__}: {context}")
                    return True, None
            except Exception as strategy_error:
                self.logger.error(f"恢复策略异常 {strategy.__class__.__name__}: {strategy_error}")
        
        # 降级处理
        self.degradation_manager.mark_service_unhealthy(context)
        self.logger.error(f"错误未处理，启动降级: {context}")
        
        return False, None
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        return {
            "degradation_level": self.degradation_manager.get_current_level().value,
            "service_states": dict(self.degradation_manager.service_states),
            "error_statistics": self.error_monitor.get_error_statistics(),
            "timestamp": datetime.now().isoformat()
        }


# 装饰器
def error_handler(error_category: ErrorCategory = ErrorCategory.SYSTEM,
                 error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 fallback_return: Any = None,
                 max_retries: int = 0):
    """错误处理装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = f"{func.__module__}.{func.__name__}"
            global_handler = get_global_error_handler()
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as error:
                    if attempt == max_retries:
                        # 最后一次尝试，使用错误处理器
                        handled, result = await global_handler.handle_error(
                            error, context, error_category, error_severity
                        )
                        if handled:
                            return result
                        else:
                            return fallback_return
                    else:
                        await asyncio.sleep(2 ** attempt)  # 指数退避
            return fallback_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = f"{func.__module__}.{func.__name__}"
            global_handler = get_global_error_handler()
            
            try:
                return func(*args, **kwargs)
            except Exception as error:
                # 同步版本只记录错误，不进行异步处理
                global_handler.error_monitor.record_error(error, context, error_category, error_severity)
                return fallback_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def graceful_degradation(service_name: str, fallback_func: Callable = None):
    """优雅降级装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            global_handler = get_global_error_handler()
            degradation_manager = global_handler.degradation_manager
            
            try:
                return await func(*args, **kwargs)
            except Exception as error:
                # 标记服务不健康
                degradation_manager.mark_service_unhealthy(service_name)
                
                # 使用回退函数
                if fallback_func:
                    try:
                        return await fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"回退函数执行失败: {fallback_error}")
                
                # 返回默认值
                return None
        
        return wrapper
    return decorator


# 全局错误处理器
_global_error_handler = None


def initialize_error_handling() -> GlobalErrorHandler:
    """初始化全局错误处理"""
    global _global_error_handler
    _global_error_handler = GlobalErrorHandler()
    logger.info("全局错误处理初始化完成")
    return _global_error_handler


def get_global_error_handler() -> GlobalErrorHandler:
    """获取全局错误处理器"""
    if _global_error_handler is None:
        raise RuntimeError("全局错误处理器未初始化")
    return _global_error_handler


if __name__ == "__main__":
    # 示例用法
    async def main():
        # 初始化错误处理
        error_handler = initialize_error_handling()
        
        # 添加自定义错误处理器
        fallback_handler = FallbackErrorHandler({
            "database_query": {"status": "error", "data": []},
            "external_api": {"status": "unavailable", "default": True}
        })
        error_handler.add_error_handler(fallback_handler)
        
        # 添加恢复策略
        async def health_check():
            # 模拟健康检查
            return True
        
        health_strategy = HealthCheckRecoveryStrategy(health_check)
        error_handler.add_recovery_strategy(health_strategy)
        
        # 测试错误处理
        @error_handler(error_category=ErrorCategory.DATABASE, fallback_return={"status": "error"})
        async def risky_operation():
            # 模拟可能失败的操作
            import random
            if random.random() < 0.7:  # 70%概率失败
                raise Exception("数据库连接失败")
            return {"status": "success", "data": "some data"}
        
        # 执行多次测试
        for i in range(10):
            try:
                result = await risky_operation()
                print(f"操作 {i+1} 结果: {result}")
            except Exception as e:
                print(f"操作 {i+1} 异常: {e}")
        
        # 获取系统健康状态
        health = error_handler.get_system_health()
        print(f"系统健康状态: {health}")
        
        # 获取错误统计
        stats = error_handler.error_monitor.get_error_statistics()
        print(f"错误统计: {stats}")
    
    # 运行示例
    asyncio.run(main())