"""
Performance Monitoring System for Zanalytics
Tracks system metrics, query performance, and resource usage
"""

import time
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import json

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: datetime
    metric_type: str
    component: str
    value: float
    metadata: Dict[str, Any]

class PerformanceMonitor:
    """
    Monitors and tracks system performance metrics
    """
    
    def __init__(self, log_file: str = "performance.log"):
        self.log_file = log_file
        self.metrics: List[PerformanceMetric] = []
        self.active_timers: Dict[str, float] = {}
        
        # Setup logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        self.active_timers[operation] = time.time()
        
    def end_timer(self, operation: str, component: str = "general", 
                  metadata: Optional[Dict] = None) -> float:
        """End timing and record the metric"""
        if operation not in self.active_timers:
            self.logger.warning(f"Timer {operation} was not started")
            return 0.0
            
        elapsed = time.time() - self.active_timers[operation]
        del self.active_timers[operation]
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type="execution_time",
            component=component,
            value=elapsed,
            metadata=metadata or {}
        )
        
        self.record_metric(metric)
        return elapsed
        
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric"""
        self.metrics.append(metric)
        self.logger.info(f"Metric recorded: {json.dumps(asdict(metric), default=str)}")
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "active_connections": len(psutil.net_connections()),
        }
        
    def record_system_snapshot(self, component: str = "system") -> None:
        """Record current system state"""
        metrics = self.get_system_metrics()
        
        for metric_name, value in metrics.items():
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type="system_resource",
                component=component,
                value=value,
                metadata={"metric_name": metric_name}
            )
            self.record_metric(metric)
            
    def get_performance_summary(self, 
                               component: Optional[str] = None,
                               metric_type: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for recorded metrics"""
        filtered_metrics = self.metrics
        
        if component:
            filtered_metrics = [m for m in filtered_metrics if m.component == component]
        if metric_type:
            filtered_metrics = [m for m in filtered_metrics if m.metric_type == metric_type]
            
        if not filtered_metrics:
            return {"message": "No metrics found"}
            
        values = [m.value for m in filtered_metrics]
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "total": sum(values),
            "latest": filtered_metrics[-1].value if filtered_metrics else None
        }
        
    def export_metrics(self, output_file: str) -> None:
        """Export all metrics to JSON file"""
        metrics_data = [asdict(m) for m in self.metrics]
        
        with open(output_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
            
        self.logger.info(f"Exported {len(metrics_data)} metrics to {output_file}")

# Context manager for automatic timing
class TimedOperation:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: PerformanceMonitor, operation: str, 
                 component: str = "general", metadata: Optional[Dict] = None):
        self.monitor = monitor
        self.operation = operation
        self.component = component
        self.metadata = metadata
        
    def __enter__(self):
        self.monitor.start_timer(self.operation)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.end_timer(self.operation, self.component, self.metadata)
