"""Deployment configuration and scripts for Zanalytics"""

import os
import yaml
from typing import Dict, Any


def generate_docker_compose() -> str:
    """Generate Docker Compose configuration"""
    compose_config: Dict[str, Any] = {
        'version': '3.8',
        'services': {
            'zanalytics-api': {
                'build': {
                    'context': '.',
                    'dockerfile': 'Dockerfile'
                },
                'ports': ['8000:8000'],
                'environment': [
                    'ENVIRONMENT=production',
                    'DATABASE_URL=${DATABASE_URL}',
                    'REDIS_URL=${REDIS_URL}'
                ],
                'volumes': [
                    './data:/app/data',
                    './logs:/app/logs'
                ],
                'depends_on': ['redis', 'postgres']
            },
            'redis': {
                'image': 'redis:alpine',
                'ports': ['6379:6379'],
                'volumes': ['redis-data:/data']
            },
            'postgres': {
                'image': 'postgres:13',
                'environment': [
                    'POSTGRES_DB=zanalytics',
                    'POSTGRES_USER=${POSTGRES_USER}',
                    'POSTGRES_PASSWORD=${POSTGRES_PASSWORD}'
                ],
                'volumes': ['postgres-data:/var/lib/postgresql/data'],
                'ports': ['5432:5432']
            },
            'nginx': {
                'image': 'nginx:alpine',
                'ports': ['80:80', '443:443'],
                'volumes': [
                    './nginx.conf:/etc/nginx/nginx.conf',
                    './ssl:/etc/nginx/ssl'
                ],
                'depends_on': ['zanalytics-api']
            }
        },
        'volumes': {
            'redis-data': {},
            'postgres-data': {}
        }
    }
    return yaml.dump(compose_config, default_flow_style=False)


def generate_dockerfile() -> str:
    """Generate Dockerfile for the application"""
    dockerfile = """# Zanalytics Dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/logs /app/cache

ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run.py"]
"""
    return dockerfile


def generate_kubernetes_config() -> str:
    """Generate Kubernetes deployment configuration"""
    k8s_config: Dict[str, Any] = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': 'zanalytics',
            'labels': {'app': 'zanalytics'}
        },
        'spec': {
            'replicas': 3,
            'selector': {
                'matchLabels': {'app': 'zanalytics'}
            },
            'template': {
                'metadata': {
                    'labels': {'app': 'zanalytics'}
                },
                'spec': {
                    'containers': [{
                        'name': 'zanalytics',
                        'image': 'zanalytics:latest',
                        'ports': [{'containerPort': 8000}],
                        'env': [
                            {'name': 'ENVIRONMENT', 'value': 'production'},
                            {
                                'name': 'DATABASE_URL',
                                'valueFrom': {
                                    'secretKeyRef': {
                                        'name': 'zanalytics-secrets',
                                        'key': 'database-url'
                                    }
                                }
                            }
                        ],
                        'resources': {
                            'requests': {'memory': '256Mi', 'cpu': '250m'},
                            'limits': {'memory': '512Mi', 'cpu': '500m'}
                        },
                        'livenessProbe': {
                            'httpGet': {'path': '/health', 'port': 8000},
                            'initialDelaySeconds': 30,
                            'periodSeconds': 10
                        },
                        'readinessProbe': {
                            'httpGet': {'path': '/ready', 'port': 8000},
                            'initialDelaySeconds': 5,
                            'periodSeconds': 5
                        }
                    }]
                }
            }
        }
    }
    return yaml.dump(k8s_config, default_flow_style=False)


def main() -> None:
    """Generate deployment files"""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(output_dir, 'docker-compose.yml'), 'w') as f:
        f.write(generate_docker_compose())
    with open(os.path.join(output_dir, 'Dockerfile'), 'w') as f:
        f.write(generate_dockerfile())
    with open(os.path.join(output_dir, 'k8s_deployment.yaml'), 'w') as f:
        f.write(generate_kubernetes_config())


if __name__ == '__main__':
    main()
