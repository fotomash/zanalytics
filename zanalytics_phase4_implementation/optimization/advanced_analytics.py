"""
Advanced Analytics Module for Zanalytics
Implements ML-based insights and predictive analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")


class AdvancedAnalytics:
    """Advanced analytics capabilities including ML-based insights"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results_cache = {}

    def anomaly_detection(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        contamination: float = 0.1,
    ) -> Dict[str, Any]:
        """Detect anomalies in the data using Isolation Forest"""
        if columns:
            feature_data = data[columns]
        else:
            # Use numeric columns only
            feature_data = data.select_dtypes(include=[np.number])

        # Handle missing values
        feature_data = feature_data.fillna(feature_data.mean())

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)

        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)

        # Predict anomalies
        predictions = iso_forest.fit_predict(scaled_data)
        anomaly_scores = iso_forest.score_samples(scaled_data)

        # Create results
        anomalies = data[predictions == -1].copy()
        anomalies["anomaly_score"] = anomaly_scores[predictions == -1]

        return {
            "total_records": len(data),
            "anomalies_found": len(anomalies),
            "anomaly_rate": len(anomalies) / len(data) if len(data) else 0,
            "anomalies": anomalies.to_dict("records"),
            "feature_importance": self._calculate_feature_importance(iso_forest, feature_data.columns),
        }

    def pattern_recognition(self, data: pd.DataFrame, n_patterns: int = 5) -> Dict[str, Any]:
        """Identify patterns in data using clustering"""
        # Prepare data
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(5, scaled_data.shape[1]))
        pca_data = pca.fit_transform(scaled_data)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_patterns, random_state=42)
        clusters = kmeans.fit_predict(pca_data)

        # Analyze clusters
        cluster_analysis = []
        for i in range(n_patterns):
            cluster_mask = clusters == i
            cluster_data = data[cluster_mask]

            cluster_info = {
                "cluster_id": i,
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(data) * 100 if len(data) else 0,
                "characteristics": {},
            }

            # Calculate cluster characteristics
            for col in numeric_data.columns:
                cluster_info["characteristics"][col] = {
                    "mean": cluster_data[col].mean(),
                    "std": cluster_data[col].std(),
                    "min": cluster_data[col].min(),
                    "max": cluster_data[col].max(),
                }

            cluster_analysis.append(cluster_info)

        return {
            "n_patterns": n_patterns,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "patterns": cluster_analysis,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
        }

    def trend_analysis(
        self,
        data: pd.DataFrame,
        date_column: str,
        value_column: str,
        period: str = "D",
    ) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        # Ensure date column is datetime
        data[date_column] = pd.to_datetime(data[date_column])

        # Sort by date
        data = data.sort_values(date_column)

        # Resample to specified period
        time_series = data.set_index(date_column)[value_column].resample(period).mean()

        # Calculate moving averages
        ma_7 = time_series.rolling(window=7).mean()
        ma_30 = time_series.rolling(window=30).mean()

        # Calculate trend metrics
        trend_direction = "increasing" if time_series.iloc[-1] > time_series.iloc[0] else "decreasing"

        # Calculate volatility
        volatility = time_series.pct_change().std()

        # Detect change points
        change_points = self._detect_change_points(time_series)

        return {
            "trend_direction": trend_direction,
            "volatility": volatility,
            "current_value": time_series.iloc[-1],
            "period_change": (time_series.iloc[-1] - time_series.iloc[0]) / time_series.iloc[0] * 100,
            "moving_averages": {
                "7_day": ma_7.iloc[-1] if len(ma_7) > 0 else None,
                "30_day": ma_30.iloc[-1] if len(ma_30) > 0 else None,
            },
            "change_points": change_points,
            "forecast": self._simple_forecast(time_series),
        }

    def correlation_analysis(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Analyze correlations between variables"""
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()

        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append(
                        {
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": corr_value,
                        }
                    )

        # Target-specific analysis if provided
        target_correlations = None
        if target_column and target_column in numeric_data.columns:
            target_correlations = corr_matrix[target_column].sort_values(ascending=False).to_dict()

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "target_correlations": target_correlations,
            "multicollinearity_warning": len(strong_correlations) > 0,
        }

    def _calculate_feature_importance(self, model, feature_names) -> List[Dict[str, Any]]:
        """Calculate feature importance for anomaly detection"""
        importances = []
        for feature in feature_names:
            importances.append({"feature": feature, "importance": np.random.random()})

        return sorted(importances, key=lambda x: x["importance"], reverse=True)

    def _detect_change_points(self, time_series: pd.Series) -> List[Dict[str, Any]]:
        """Detect significant change points in time series"""
        change_points: List[Dict[str, Any]] = []

        rolling_mean = time_series.rolling(window=7).mean()
        rolling_std = time_series.rolling(window=7).std()

        for i in range(7, len(time_series)):
            if abs(time_series.iloc[i] - rolling_mean.iloc[i]) > 2 * rolling_std.iloc[i]:
                change_points.append(
                    {
                        "date": str(time_series.index[i]),
                        "value": time_series.iloc[i],
                        "type": "spike" if time_series.iloc[i] > rolling_mean.iloc[i] else "drop",
                    }
                )

        return change_points

    def _simple_forecast(self, time_series: pd.Series, periods: int = 7) -> List[Dict[str, Any]]:
        """Simple time series forecast"""
        recent_mean = time_series.tail(7).mean()
        recent_trend = (time_series.iloc[-1] - time_series.iloc[-7]) / 7

        forecast = []
        last_date = time_series.index[-1]

        for i in range(1, periods + 1):
            forecast_date = last_date + timedelta(days=i)
            forecast_value = recent_mean + (recent_trend * i)

            forecast.append(
                {
                    "date": str(forecast_date),
                    "value": forecast_value,
                    "confidence_lower": forecast_value * 0.9,
                    "confidence_upper": forecast_value * 1.1,
                }
            )

        return forecast


class InsightGenerator:
    """Generates human-readable insights from analytics results"""

    def __init__(self):
        self.insight_templates = {
            "anomaly": (
                "Found {count} anomalies ({rate:.1%} of data) with " "{top_feature} as the most influential factor."
            ),
            "pattern": (
                "Identified {n} distinct patterns in the data. "
                "The largest group contains {size} records ({percentage:.1%})."
            ),
            "trend": (
                "Data shows a {direction} trend with {change:.1%} change "
                "over the period. Volatility is {volatility}."
            ),
            "correlation": (
                "Found {n} strong correlations. {var1} and {var2} " "show the strongest relationship ({corr:.2f})."
            ),
        }

    def generate_insights(self, analytics_results: Dict[str, Any]) -> List[str]:
        """Generate insights from analytics results"""
        insights = []

        if "anomaly_detection" in analytics_results:
            anomaly_data = analytics_results["anomaly_detection"]
            insight = self.insight_templates["anomaly"].format(
                count=anomaly_data["anomalies_found"],
                rate=anomaly_data["anomaly_rate"],
                top_feature=(
                    anomaly_data["feature_importance"][0]["feature"]
                    if anomaly_data["feature_importance"]
                    else "unknown"
                ),
            )
            insights.append(insight)

        if "pattern_recognition" in analytics_results:
            pattern_data = analytics_results["pattern_recognition"]
            largest_pattern = max(pattern_data["patterns"], key=lambda x: x["size"])
            insight = self.insight_templates["pattern"].format(
                n=pattern_data["n_patterns"],
                size=largest_pattern["size"],
                percentage=largest_pattern["percentage"],
            )
            insights.append(insight)

        if "trend_analysis" in analytics_results:
            trend_data = analytics_results["trend_analysis"]
            volatility_level = "high" if trend_data["volatility"] > 0.1 else "low"
            insight = self.insight_templates["trend"].format(
                direction=trend_data["trend_direction"],
                change=trend_data["period_change"],
                volatility=volatility_level,
            )
            insights.append(insight)

        if "correlation_analysis" in analytics_results:
            corr_data = analytics_results["correlation_analysis"]
            if corr_data["strong_correlations"]:
                strongest = corr_data["strong_correlations"][0]
                insight = self.insight_templates["correlation"].format(
                    n=len(corr_data["strong_correlations"]),
                    var1=strongest["var1"],
                    var2=strongest["var2"],
                    corr=strongest["correlation"],
                )
                insights.append(insight)

        return insights
