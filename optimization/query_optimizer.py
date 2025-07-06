"""
Query Optimization System for Zanalytics
Optimizes data queries and analysis operations
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class QueryPlan:
    """Represents an optimized query execution plan"""
    original_query: str
    optimized_query: str
    execution_steps: List[Dict[str, Any]]
    estimated_cost: float
    optimizations_applied: List[str]


class QueryOptimizer:
    """Optimizes queries for better performance"""

    def __init__(self):
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.optimization_rules = [
            self._optimize_filter_pushdown,
            self._optimize_column_selection,
            self._optimize_join_order,
            self._optimize_aggregation,
            self._optimize_caching,
        ]

    def optimize_query(self, query: str, context: Optional[Dict] = None) -> QueryPlan:
        """Optimize a query and return execution plan"""
        context = context or {}

        query_structure = self._parse_query(query)
        optimized_structure = query_structure.copy()
        optimizations_applied = []

        for rule in self.optimization_rules:
            result = rule(optimized_structure, context)
            if result['applied']:
                optimizations_applied.append(result['name'])
                optimized_structure = result['structure']

        execution_steps = self._generate_execution_plan(optimized_structure)
        estimated_cost = self._estimate_cost(execution_steps, context)
        optimized_query = self._build_query(optimized_structure)

        return QueryPlan(
            original_query=query,
            optimized_query=optimized_query,
            execution_steps=execution_steps,
            estimated_cost=estimated_cost,
            optimizations_applied=optimizations_applied,
        )

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query into structured format"""
        structure = {
            'type': 'select',
            'columns': ['*'],
            'from': [],
            'where': [],
            'join': [],
            'group_by': [],
            'order_by': [],
            'limit': None,
        }

        ql = query.lower()
        if 'select' in ql:
            select_idx = ql.find('select')
            from_idx = ql.find('from')
            if select_idx >= 0 and from_idx > select_idx:
                cols = query[select_idx + 6:from_idx].strip()
                structure['columns'] = [c.strip() for c in cols.split(',')]

        if 'from' in ql:
            from_idx = ql.find('from')
            where_idx = ql.find('where') if 'where' in ql else len(query)
            if from_idx >= 0:
                tables = query[from_idx + 4:where_idx].strip()
                structure['from'] = [t.strip() for t in tables.split(',')]
        return structure

    def _optimize_filter_pushdown(self, structure: Dict, context: Dict) -> Dict[str, Any]:
        applied = False
        if structure.get('where') and structure.get('join'):
            applied = True
        return {'applied': applied, 'name': 'filter_pushdown', 'structure': structure}

    def _optimize_column_selection(self, structure: Dict, context: Dict) -> Dict[str, Any]:
        applied = False
        if '*' in structure.get('columns', []):
            structure['columns'] = ['id', 'name', 'value']
            applied = True
        return {'applied': applied, 'name': 'column_selection', 'structure': structure}

    def _optimize_join_order(self, structure: Dict, context: Dict) -> Dict[str, Any]:
        applied = False
        if len(structure.get('join', [])) > 1:
            applied = True
        return {'applied': applied, 'name': 'join_reordering', 'structure': structure}

    def _optimize_aggregation(self, structure: Dict, context: Dict) -> Dict[str, Any]:
        applied = False
        if structure.get('group_by'):
            applied = True
        return {'applied': applied, 'name': 'aggregation_optimization', 'structure': structure}

    def _optimize_caching(self, structure: Dict, context: Dict) -> Dict[str, Any]:
        applied = False
        qhash = hash(str(structure))
        if qhash in self.query_stats and self.query_stats[qhash].get('count', 0) > 5:
            structure['cache_hint'] = True
            applied = True
        return {'applied': applied, 'name': 'caching_optimization', 'structure': structure}

    def _generate_execution_plan(self, structure: Dict) -> List[Dict[str, Any]]:
        steps = []
        if structure.get('from'):
            steps.append({'step': 'scan', 'tables': structure['from'], 'columns': structure.get('columns', ['*'])})
        if structure.get('where'):
            steps.append({'step': 'filter', 'conditions': structure['where']})
        if structure.get('join'):
            steps.append({'step': 'join', 'joins': structure['join']})
        if structure.get('group_by'):
            steps.append({'step': 'aggregate', 'group_by': structure['group_by']})
        if structure.get('order_by'):
            steps.append({'step': 'sort', 'order_by': structure['order_by']})
        if structure.get('limit'):
            steps.append({'step': 'limit', 'limit': structure['limit']})
        return steps

    def _estimate_cost(self, steps: List[Dict], context: Dict) -> float:
        cost = 0.0
        for step in steps:
            if step['step'] == 'scan':
                cost += 10.0
            elif step['step'] == 'filter':
                cost += 2.0
            elif step['step'] == 'join':
                cost += 20.0
            elif step['step'] == 'aggregate':
                cost += 15.0
            elif step['step'] == 'sort':
                cost += 5.0
        return cost

    def _build_query(self, structure: Dict) -> str:
        parts = []
        if structure.get('columns'):
            parts.append(f"SELECT {', '.join(structure['columns'])}")
        if structure.get('from'):
            parts.append(f"FROM {', '.join(structure['from'])}")
        if structure.get('where'):
            parts.append(f"WHERE {' AND '.join(structure['where'])}")
        return ' '.join(parts)

    def record_query_execution(self, query: str, execution_time: float) -> None:
        qhash = hash(query)
        if qhash not in self.query_stats:
            self.query_stats[qhash] = {'query': query, 'count': 0, 'total_time': 0.0, 'avg_time': 0.0}
        stats = self.query_stats[qhash]
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
