"""
Windows数据加载器性能监控和错误处理

提供详细的性能监控、错误处理和自动调整功能。
"""

import os
import sys
import time
import threading
import queue
import traceback
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import warnings

import numpy as np
import psutil
import torch

from ..Tools.utils import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    INFO = "info"        # 信息性错误，不影响功能
    WARNING = "warning"  # 警告，功能可能受限
    ERROR = "error"      # 错误，功能部分失效
    CRITICAL = "critical"  # 严重错误，功能完全失效


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    batch_load_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    throughput_imgs_per_sec: Optional[float] = None
    batch_size: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'timestamp': self.timestamp,
            'batch_load_time_ms': self.batch_load_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_percent': self.cpu_percent,
            'batch_size': self.batch_size
        }
        
        if self.gpu_memory_mb is not None:
            result['gpu_memory_mb'] = self.gpu_memory_mb
        
        if self.throughput_imgs_per_sec is not None:
            result['throughput_imgs_per_sec'] = self.throughput_imgs_per_sec
        
        return result


@dataclass
class ErrorRecord:
    """错误记录"""
    timestamp: float
    severity: ErrorSeverity
    error_type: str
    message: str
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'error_type': self.error_type,
            'message': self.message,
            'traceback': self.traceback,
            'context': self.context
        }


class WindowsPerformanceMonitor:
    """Windows性能监控器"""
    
    def __init__(self, 
                 monitor_interval: float = 1.0,
                 history_size: int = 1000,
                 log_dir: str = "./logs/windows_loader"):
        """
        参数:
            monitor_interval: 监控间隔（秒）
            history_size: 历史记录大小
            log_dir: 日志目录
        """
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        self.log_dir = log_dir
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 性能指标历史
        self.performance_history: List[PerformanceMetrics] = []
        self.error_history: List[ErrorRecord] = []
        
        # 监控线程
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self.stats = {
            'total_batches': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'start_time': time.time(),
            'last_batch_time': None
        }
        
        # 性能阈值
        self.thresholds = {
            'max_batch_load_time_ms': 1000.0,  # 最大批次加载时间
            'max_memory_mb': 8192.0,           # 最大内存使用
            'max_cpu_percent': 90.0,           # 最大CPU使用率
            'error_rate_threshold': 0.1        # 错误率阈值（错误数/批次数）
        }
        
        logger.info(f"Windows性能监控器初始化完成，日志目录: {log_dir}")
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            logger.warning("监控已在运行中")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="windows_performance_monitor"
        )
        self.monitor_thread.start()
        
        logger.info("Windows性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Windows性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集系统指标
                self._collect_system_metrics()
                
                # 检查性能问题
                self._check_performance_issues()
                
                # 保存历史记录
                self._cleanup_history()
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
            
            # 等待下一个监控周期
            time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # 获取进程信息
            process = psutil.Process()
            
            # CPU使用率
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # 内存使用
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # GPU内存使用（如果可用）
            gpu_memory_mb = None
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            # 创建指标记录
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                batch_load_time_ms=0,  # 将在记录批次时更新
                memory_usage_mb=memory_mb,
                cpu_percent=cpu_percent,
                gpu_memory_mb=gpu_memory_mb
            )
            
            # 添加到历史记录
            self.performance_history.append(metrics)
            
        except Exception as e:
            self.record_error(
                ErrorSeverity.WARNING,
                "SystemMetricsError",
                f"收集系统指标失败: {e}"
            )
    
    def record_batch_performance(self, load_time_ms: float, batch_size: int = 1):
        """
        记录批次性能
        
        参数:
            load_time_ms: 批次加载时间（毫秒）
            batch_size: 批次大小
        """
        try:
            # 更新统计信息
            self.stats['total_batches'] += 1
            self.stats['last_batch_time'] = time.time()
            
            # 计算吞吐量
            throughput = None
            if load_time_ms > 0:
                throughput = (batch_size * 1000) / load_time_ms  # 图像/秒
            
            # 获取最新的性能指标并更新
            if self.performance_history:
                latest = self.performance_history[-1]
                latest.batch_load_time_ms = load_time_ms
                latest.batch_size = batch_size
                latest.throughput_imgs_per_sec = throughput
            
            # 检查性能问题
            self._check_batch_performance(load_time_ms, batch_size)
            
        except Exception as e:
            self.record_error(
                ErrorSeverity.WARNING,
                "BatchPerformanceError",
                f"记录批次性能失败: {e}"
            )
    
    def _check_batch_performance(self, load_time_ms: float, batch_size: int):
        """检查批次性能问题"""
        warnings = []
        
        # 检查加载时间
        if load_time_ms > self.thresholds['max_batch_load_time_ms']:
            warnings.append(
                f"批次加载时间过长: {load_time_ms:.1f} ms > "
                f"{self.thresholds['max_batch_load_time_ms']} ms"
            )
        
        # 检查内存使用
        if self.performance_history:
            latest_memory = self.performance_history[-1].memory_usage_mb
            if latest_memory > self.thresholds['max_memory_mb']:
                warnings.append(
                    f"内存使用过高: {latest_memory:.1f} MB > "
                    f"{self.thresholds['max_memory_mb']} MB"
                )
        
        # 检查CPU使用率
        if self.performance_history:
            latest_cpu = self.performance_history[-1].cpu_percent
            if latest_cpu > self.thresholds['max_cpu_percent']:
                warnings.append(
                    f"CPU使用率过高: {latest_cpu:.1f}% > "
                    f"{self.thresholds['max_cpu_percent']}%"
                )
        
        # 记录警告
        for warning in warnings:
            self.record_error(
                ErrorSeverity.WARNING,
                "PerformanceWarning",
                warning,
                context={
                    'load_time_ms': load_time_ms,
                    'batch_size': batch_size,
                    'thresholds': self.thresholds.copy()
                }
            )
    
    def _check_performance_issues(self):
        """检查性能问题"""
        # 检查错误率
        if self.stats['total_batches'] > 0:
            error_rate = self.stats['total_errors'] / self.stats['total_batches']
            if error_rate > self.thresholds['error_rate_threshold']:
                self.record_error(
                    ErrorSeverity.ERROR,
                    "HighErrorRate",
                    f"错误率过高: {error_rate:.2%} > "
                    f"{self.thresholds['error_rate_threshold']:.0%}",
                    context={
                        'total_batches': self.stats['total_batches'],
                        'total_errors': self.stats['total_errors'],
                        'error_rate': error_rate
                    }
                )
    
    def record_error(self, 
                     severity: ErrorSeverity,
                     error_type: str,
                     message: str,
                     traceback_str: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None):
        """
        记录错误
        
        参数:
            severity: 错误严重程度
            error_type: 错误类型
            message: 错误消息
            traceback_str: 堆栈跟踪
            context: 上下文信息
        """
        # 更新统计信息
        if severity == ErrorSeverity.ERROR:
            self.stats['total_errors'] += 1
        elif severity == ErrorSeverity.WARNING:
            self.stats['total_warnings'] += 1
        
        # 创建错误记录
        error_record = ErrorRecord(
            timestamp=time.time(),
            severity=severity,
            error_type=error_type,
            message=message,
            traceback=traceback_str,
            context=context or {}
        )
        
        # 添加到历史记录
        self.error_history.append(error_record)
        
        # 记录到日志
        log_message = f"[{severity.value.upper()}] {error_type}: {message}"
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # 如果有关键错误，触发警报
        if severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            self._trigger_alert(error_record)
    
    def _trigger_alert(self, error_record: ErrorRecord):
        """触发警报"""
        # 这里可以实现警报逻辑，例如发送邮件、写入特殊日志文件等
        alert_file = os.path.join(self.log_dir, "alerts.json")
        
        try:
            alert_data = error_record.to_dict()
            alert_data['alert_time'] = datetime.now().isoformat()
            
            # 读取现有警报
            existing_alerts = []
            if os.path.exists(alert_file):
                with open(alert_file, 'r', encoding='utf-8') as f:
                    existing_alerts = json.load(f)
            
            # 添加新警报
            existing_alerts.append(alert_data)
            
            # 保存警报（只保留最近100个）
            if len(existing_alerts) > 100:
                existing_alerts = existing_alerts[-100:]
            
            with open(alert_file, 'w', encoding='utf-8') as f:
                json.dump(existing_alerts, f, indent=2, ensure_ascii=False)
            
            logger.warning(f"警报已记录到: {alert_file}")
            
        except Exception as e:
            logger.error(f"记录警报失败: {e}")
    
    def _cleanup_history(self):
        """清理历史记录"""
        # 清理性能历史记录
        if len(self.performance_history) > self.history_size:
            self.performance_history = self.performance_history[-self.history_size:]
        
        # 清理错误历史记录
        if len(self.error_history) > self.history_size:
            self.error_history = self.error_history[-self.history_size:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.performance_history:
            return {'status': 'no_data'}
        
        # 计算统计信息
        load_times = [m.batch_load_time_ms for m in self.performance_history if m.batch_load_time_ms > 0]
        memory_usage = [m.memory_usage_mb for m in self.performance_history]
        cpu_usage = [m.cpu_percent for m in self.performance_history]
        
        report = {
            'monitoring_duration_s': time.time() - self.stats['start_time'],
            'total_batches': self.stats['total_batches'],
            'total_errors': self.stats['total_errors'],
            'total_warnings': self.stats['total_warnings'],
            'performance': {
                'avg_load_time_ms': np.mean(load_times) if load_times else 0,
                'max_load_time_ms': max(load_times) if load_times else 0,
                'min_load_time_ms': min(load_times) if load_times else 0,
                'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
                'max_memory_mb': max(memory_usage) if memory_usage else 0,
                'avg_cpu_percent': np.mean(cpu_usage) if cpu_usage else 0,
                'max_cpu_percent': max(cpu_usage) if cpu_usage else 0,
            },
            'error_rate': self.stats['total_errors'] / max(self.stats['total_batches'], 1),
            'recent_errors': [
                error.to_dict() 
                for error in self.error_history[-10:]  # 最近10个错误
            ],
            'recommendations': self._generate_recommendations()
        }
        
        # 添加GPU信息（如果可用）
        if torch.cuda.is_available():
            gpu_memory = [m.gpu_memory_mb for m in self.performance_history if m.gpu_memory_mb is not None]
            if gpu_memory:
                report['performance']['avg_gpu_memory_mb'] = np.mean(gpu_memory)
                report['performance']['max_gpu_memory_mb'] = max(gpu_memory)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if not self.performance_history:
            return recommendations
        
        # 分析性能数据
        load_times = [m.batch_load_time_ms for m in self.performance_history if m.batch_load_time_ms > 0]
        
        if load_times:
            avg_load_time = np.mean(load_times)
            
            # 加载时间建议
            if avg_load_time > 500:  # 大于500ms
                recommendations.append(
                    "批次加载时间过长，考虑启用数据预加载或增加缓存大小"
                )
            elif avg_load_time > 100:  # 大于100ms
                recommendations.append(
                    "批次加载时间中等，考虑使用线程池优化I/O操作"
                )
        
        # 内存使用建议
        memory_usage = [m.memory_usage_mb for m in self.performance_history]
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            
            if avg_memory > 4096:  # 大于4GB
                recommendations.append(
                    "内存使用较高，考虑启用内存映射文件或减少缓存大小"
                )
            elif avg_memory > 2048:  # 大于2GB
                recommendations.append(
                    "内存使用中等，考虑监控内存泄漏或优化数据格式"
                )
        
        # 错误率建议
        error_rate = self.stats['total_errors'] / max(self.stats['total_batches'], 1)
        if error_rate > 0.05:  # 错误率大于5%
            recommendations.append(
                f"错误率较高 ({error_rate:.1%})，建议检查数据源和网络连接"
            )
        
        return recommendations
    
    def save_report(self, filename: Optional[str] = None):
        """保存报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"windows_loader_report_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            report = self.get_performance_report()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"性能报告已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存性能报告失败: {e}")
            return None
    
    def export_history(self, export_dir: Optional[str] = None):
        """导出历史数据"""
        if export_dir is None:
            export_dir = os.path.join(self.log_dir, "exports")
        
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出性能历史
        perf_file = os.path.join(export_dir, f"performance_history_{timestamp}.json")
        try:
            perf_data = [metrics.to_dict() for metrics in self.performance_history]
            with open(perf_file, 'w', encoding='utf-8') as f:
                json.dump(perf_data, f, indent=2, ensure_ascii=False)
            logger.info(f"性能历史已导出到: {perf_file}")
        except Exception as e:
            logger.error(f"导出性能历史失败: {e}")
        
        # 导出错误历史
        error_file = os.path.join(export_dir, f"error_history_{timestamp}.json")
        try:
            error_data = [error.to_dict() for error in self.error_history]
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            logger.info(f"错误历史已导出到: {error_file}")
        except Exception as e:
            logger.error(f"导出错误历史失败: {e}")
        
        return perf_file, error_file
    
    def reset(self):
        """重置监控器"""
        self.performance_history.clear()
        self.error_history.clear()
        
        self.stats = {
            'total_batches': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'start_time': time.time(),
            'last_batch_time': None
        }
        
        logger.info("性能监控器已重置")


class WindowsErrorHandler:
    """Windows错误处理器"""
    
    def __init__(self, performance_monitor: Optional[WindowsPerformanceMonitor] = None):
        """
        参数:
            performance_monitor: 性能监控器实例
        """
        self.performance_monitor = performance_monitor
        self.error_handlers = {
            'serialization': self._handle_serialization_error,
            'memory': self._handle_memory_error,
            'io': self._handle_io_error,
            'timeout': self._handle_timeout_error,
            'unknown': self._handle_unknown_error
        }
        
        # 错误恢复策略
        self.recovery_strategies = {
            'serialization': ['preload', 'memmap', 'fallback'],
            'memory': ['memmap', 'stream', 'reduce_batch'],
            'io': ['threadpool', 'cache', 'retry'],
            'timeout': ['reduce_workers', 'increase_timeout', 'fallback']
        }
    
    def handle_error(self,
                     error: Exception,
                     context: Optional[Dict[str, Any]] = None,
                     severity: ErrorSeverity = ErrorSeverity.ERROR) -> Dict[str, Any]:
        """
        处理错误
        
        参数:
            error: 异常对象
            context: 上下文信息
            severity: 错误严重程度
            
        返回:
            处理结果
        """
        # 确定错误类型
        error_type = self._classify_error(error)
        
        # 记录错误
        if self.performance_monitor:
            self.performance_monitor.record_error(
                severity=severity,
                error_type=error_type,
                message=str(error),
                traceback_str=traceback.format_exc(),
                context=context
            )
        
        # 调用错误处理器
        handler = self.error_handlers.get(error_type, self.error_handlers['unknown'])
        result = handler(error, context, severity)
        
        return result
    
    def _classify_error(self, error: Exception) -> str:
        """分类错误"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # 序列化错误
        if any(keyword in error_str for keyword in ['pickle', 'serializ', 'marshal']):
            return 'serialization'
        
        # 内存错误
        if any(keyword in error_str for keyword in ['memory', 'out of memory', 'oom']):
            return 'memory'
        
        # I/O错误
        if any(keyword in error_str for keyword in ['file', 'io', 'disk', 'permission']):
            return 'io'
        
        # 超时错误
        if any(keyword in error_str for keyword in ['timeout', 'timed out']):
            return 'timeout'
        
        # 根据错误类型分类
        if 'Serialization' in error_type or 'Pickle' in error_type:
            return 'serialization'
        elif 'Memory' in error_type:
            return 'memory'
        elif any(io_type in error_type for io_type in ['IO', 'File', 'OS']):
            return 'io'
        elif 'Timeout' in error_type:
            return 'timeout'
        
        return 'unknown'
    
    def _handle_serialization_error(self,
                                   error: Exception,
                                   context: Optional[Dict[str, Any]],
                                   severity: ErrorSeverity) -> Dict[str, Any]:
        """处理序列化错误"""
        logger.warning("检测到序列化错误，Windows多进程可能受限")
        
        recommendations = [
            "Windows系统上多进程需要可序列化的数据集",
            "考虑使用预加载策略将数据完全加载到内存",
            "或使用内存映射文件避免序列化问题",
            "检查数据集中是否包含不可序列化的对象（如文件句柄、lambda函数等）"
        ]
        
        return {
            'error_type': 'serialization',
            'handled': True,
            'recovery_strategies': self.recovery_strategies['serialization'],
            'recommendations': recommendations,
            'fallback_suggestion': '使用单进程数据加载器或启用预加载'
        }
    
    def _handle_memory_error(self,
                            error: Exception,
                            context: Optional[Dict[str, Any]],
                            severity: ErrorSeverity) -> Dict[str, Any]:
        """处理内存错误"""
        logger.error("检测到内存错误，可能需要优化内存使用")
        
        recommendations = [
            "减少批次大小以降低内存需求",
            "启用内存映射文件处理大数据集",
            "使用流式处理逐块加载数据",
            "清理不必要的缓存和临时数据"
        ]
        
        return {
            'error_type': 'memory',
            'handled': True,
            'recovery_strategies': self.recovery_strategies['memory'],
            'recommendations': recommendations,
            'fallback_suggestion': '减少批次大小或启用内存映射'
        }
    
    def _handle_io_error(self,
                        error: Exception,
                        context: Optional[Dict[str, Any]],
                        severity: ErrorSeverity) -> Dict[str, Any]:
        """处理I/O错误"""
        logger.warning("检测到I/O错误，数据访问可能受限")
        
        recommendations = [
            "检查文件路径和权限",
            "使用线程池优化I/O操作",
            "启用数据缓存减少磁盘访问",
            "考虑使用更快的存储设备"
        ]
        
        return {
            'error_type': 'io',
            'handled': True,
            'recovery_strategies': self.recovery_strategies['io'],
            'recommendations': recommendations,
            'fallback_suggestion': '使用线程池或启用缓存'
        }
    
    def _handle_timeout_error(self,
                             error: Exception,
                             context: Optional[Dict[str, Any]],
                             severity: ErrorSeverity) -> Dict[str, Any]:
        """处理超时错误"""
        logger.warning("检测到超时错误，操作执行时间过长")
        
        recommendations = [
            "减少工作线程数量",
            "增加超时时间限制",
            "优化数据加载逻辑",
            "检查网络或存储性能"
        ]
        
        return {
            'error_type': 'timeout',
            'handled': True,
            'recovery_strategies': self.recovery_strategies['timeout'],
            'recommendations': recommendations,
            'fallback_suggestion': '减少工作线程或增加超时时间'
        }
    
    def _handle_unknown_error(self,
                             error: Exception,
                             context: Optional[Dict[str, Any]],
                             severity: ErrorSeverity) -> Dict[str, Any]:
        """处理未知错误"""
        logger.error(f"检测到未知错误: {error}")
        
        return {
            'error_type': 'unknown',
            'handled': False,
            'recovery_strategies': ['fallback', 'retry', 'abort'],
            'recommendations': ["检查错误日志以获取更多信息", "尝试重启数据加载器"],
            'fallback_suggestion': '使用基础数据加载器或联系技术支持'
        }
    
    def get_recovery_plan(self, error_type: str) -> List[str]:
        """获取错误恢复计划"""
        return self.recovery_strategies.get(error_type, ['fallback'])


# 导出公共接口
__all__ = [
    'ErrorSeverity',
    'PerformanceMetrics',
    'ErrorRecord',
    'WindowsPerformanceMonitor',
    'WindowsErrorHandler'
]
