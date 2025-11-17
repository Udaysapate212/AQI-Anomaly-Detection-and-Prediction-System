"""
Intelligent Alert System for AQI Anomaly Detection

This module provides context-aware alert generation and management
for detected environmental anomalies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeverityClassifier:
    """
    Classify anomaly severity based on multiple factors.
    """
    
    def __init__(self, config=None):
        """
        Initialize severity classifier.
        
        Args:
            config: Configuration dictionary with thresholds
        """
        if config is None:
            config = self._get_default_config()
        
        self.config = config
        self.thresholds = config.get('severity_thresholds', {
            'critical': -0.2,
            'high': -0.15,
            'medium': -0.1,
            'low': 0.0
        })
    
    def _get_default_config(self):
        """Get default configuration."""
        return {
            'severity_thresholds': {
                'critical': -0.2,
                'high': -0.15,
                'medium': -0.1,
                'low': 0.0
            },
            'pollutant_weights': {
                'PM2.5': 2.0,
                'PM10': 1.5,
                'CO': 1.5,
                'NO2': 1.3,
                'SO2': 1.3,
                'O3': 1.2
            },
            'time_multipliers': {
                'night': 1.2,  # 10 PM - 6 AM
                'rush_hour': 1.1,  # 7-9 AM, 5-7 PM
                'normal': 1.0
            },
            'seasonal_multipliers': {
                'winter': 1.3,
                'summer': 1.0,
                'monsoon': 0.9,
                'autumn': 1.1
            }
        }
    
    def classify_severity(self, anomaly_score, percentile_90=None, context=None):
        """
        Classify anomaly severity.
        
        Args:
            anomaly_score: Anomaly score from detector (-1 for anomaly)
            percentile_90: 90th percentile of anomaly scores
            context: Additional context (time, pollutants, etc.)
            
        Returns:
            str: Severity level ('critical', 'high', 'medium', 'low')
        """
        # Adjust score based on context if provided
        adjusted_score = anomaly_score
        
        if context is not None:
            adjusted_score = self._apply_context_adjustments(anomaly_score, context)
        
        # Use percentile-based thresholds if available
        if percentile_90 is not None:
            if adjusted_score < percentile_90 * 2:
                return 'critical'
            elif adjusted_score < percentile_90 * 1.5:
                return 'high'
            elif adjusted_score < percentile_90:
                return 'medium'
            else:
                return 'low'
        
        # Use absolute thresholds
        if adjusted_score <= self.thresholds['critical']:
            return 'critical'
        elif adjusted_score <= self.thresholds['high']:
            return 'high'
        elif adjusted_score <= self.thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _apply_context_adjustments(self, score, context):
        """
        Apply context-based adjustments to anomaly score.
        
        Args:
            score: Base anomaly score
            context: Context dictionary
            
        Returns:
            float: Adjusted score
        """
        adjusted_score = score
        
        # Time of day adjustment
        if 'hour' in context:
            hour = context['hour']
            if 22 <= hour or hour <= 6:
                adjusted_score *= self.config['time_multipliers']['night']
            elif hour in [7, 8, 9, 17, 18, 19]:
                adjusted_score *= self.config['time_multipliers']['rush_hour']
        
        # Seasonal adjustment
        if 'season' in context:
            season = context['season'].lower()
            multiplier = self.config['seasonal_multipliers'].get(season, 1.0)
            adjusted_score *= multiplier
        
        # Pollutant-based adjustment
        if 'pollutants' in context:
            max_weight = 1.0
            for pollutant, value in context['pollutants'].items():
                weight = self.config['pollutant_weights'].get(pollutant, 1.0)
                if weight > max_weight:
                    max_weight = weight
            adjusted_score *= max_weight
        
        return adjusted_score
    
    def get_severity_emoji(self, severity):
        """Get emoji representation of severity."""
        emoji_map = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        return emoji_map.get(severity, '⚪')
    
    def get_severity_color(self, severity):
        """Get color code for severity."""
        color_map = {
            'critical': '#DC143C',
            'high': '#FF8C00',
            'medium': '#FFD700',
            'low': '#90EE90'
        }
        return color_map.get(severity, '#CCCCCC')


class Alert:
    """
    Represents a single anomaly alert.
    """
    
    def __init__(self, alert_id, timestamp, city, severity, anomaly_score, 
                 pollutants, description, recommendations):
        """
        Initialize alert.
        
        Args:
            alert_id: Unique alert identifier
            timestamp: When the anomaly was detected
            city: City name
            severity: Severity level
            anomaly_score: Numerical anomaly score
            pollutants: Dictionary of pollutant values
            description: Human-readable description
            recommendations: List of recommended actions
        """
        self.alert_id = alert_id
        self.timestamp = timestamp
        self.city = city
        self.severity = severity
        self.anomaly_score = anomaly_score
        self.pollutants = pollutants
        self.description = description
        self.recommendations = recommendations
        self.status = 'active'
        self.acknowledged = False
        self.acknowledged_by = None
        self.acknowledged_at = None
    
    def to_dict(self):
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'timestamp': str(self.timestamp),
            'city': self.city,
            'severity': self.severity,
            'anomaly_score': float(self.anomaly_score),
            'pollutants': self.pollutants,
            'description': self.description,
            'recommendations': self.recommendations,
            'status': self.status,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': str(self.acknowledged_at) if self.acknowledged_at else None
        }
    
    def acknowledge(self, user='system'):
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.now()
        logger.info(f"Alert {self.alert_id} acknowledged by {user}")


class AlertGenerator:
    """
    Generate alerts from detected anomalies.
    """
    
    def __init__(self):
        """Initialize alert generator."""
        self.severity_classifier = SeverityClassifier()
        self.alert_counter = 0
    
    def generate_alert(self, row, anomaly_score, percentile_90=None):
        """
        Generate alert from anomaly detection result.
        
        Args:
            row: DataFrame row with anomaly data
            anomaly_score: Anomaly score from detector
            percentile_90: 90th percentile for severity classification
            
        Returns:
            Alert: Generated alert object
        """
        # Extract context
        context = self._extract_context(row)
        
        # Classify severity
        severity = self.severity_classifier.classify_severity(
            anomaly_score, percentile_90, context
        )
        
        # Extract pollutants
        pollutants = self._extract_pollutants(row)
        
        # Generate description
        description = self._generate_description(row, severity, pollutants)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(severity, pollutants)
        
        # Create alert
        self.alert_counter += 1
        alert_id = f"ALT-{datetime.now().strftime('%Y%m%d')}-{self.alert_counter:04d}"
        
        alert = Alert(
            alert_id=alert_id,
            timestamp=row.get('Date', datetime.now()),
            city=row.get('City', 'Unknown'),
            severity=severity,
            anomaly_score=anomaly_score,
            pollutants=pollutants,
            description=description,
            recommendations=recommendations
        )
        
        return alert
    
    def _extract_context(self, row):
        """Extract context information from row."""
        context = {}
        
        # Extract time information if available
        if 'Date' in row:
            date = pd.to_datetime(row['Date'])
            context['hour'] = date.hour if hasattr(date, 'hour') else 12
            context['month'] = date.month
            context['day_of_week'] = date.dayofweek if hasattr(date, 'dayofweek') else 0
            
            # Determine season
            month = date.month
            if month in [12, 1, 2]:
                context['season'] = 'winter'
            elif month in [3, 4, 5]:
                context['season'] = 'summer'
            elif month in [6, 7, 8, 9]:
                context['season'] = 'monsoon'
            else:
                context['season'] = 'autumn'
        
        # Extract pollutants
        pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3', 'NOx', 'NH3']
        pollutants = {}
        for col in pollutant_cols:
            if col in row and not pd.isna(row[col]):
                pollutants[col] = row[col]
        context['pollutants'] = pollutants
        
        return context
    
    def _extract_pollutants(self, row):
        """Extract pollutant values from row."""
        pollutant_cols = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3', 'NOx', 'NH3', 
                          'Benzene', 'Toluene', 'Xylene']
        pollutants = {}
        
        for col in pollutant_cols:
            if col in row and not pd.isna(row[col]):
                pollutants[col] = float(row[col])
        
        return pollutants
    
    def _generate_description(self, row, severity, pollutants):
        """Generate human-readable description."""
        city = row.get('City', 'Unknown location')
        date = row.get('Date', 'Unknown date')
        
        # Find main contributing pollutants
        if pollutants:
            top_pollutants = sorted(pollutants.items(), key=lambda x: x[1], reverse=True)[:3]
            pollutant_str = ', '.join([f"{p[0]}: {p[1]:.2f}" for p in top_pollutants])
        else:
            pollutant_str = "Multiple pollutants"
        
        description = (
            f"{severity.upper()} anomaly detected in {city} on {date}. "
            f"Elevated levels: {pollutant_str}. "
            f"This represents a significant deviation from normal air quality patterns."
        )
        
        return description
    
    def _generate_recommendations(self, severity, pollutants):
        """Generate actionable recommendations."""
        recommendations = []
        
        # Severity-based recommendations
        if severity == 'critical':
            recommendations.extend([
                "⚠️ IMMEDIATE ACTION REQUIRED",
                "Issue public health advisory",
                "Consider temporary traffic restrictions",
                "Alert vulnerable populations (elderly, children, respiratory patients)",
                "Increase monitoring frequency"
            ])
        elif severity == 'high':
            recommendations.extend([
                "Increase air quality monitoring",
                "Notify local health authorities",
                "Advise sensitive groups to limit outdoor activities",
                "Investigate potential pollution sources"
            ])
        elif severity == 'medium':
            recommendations.extend([
                "Continue monitoring situation",
                "Review recent industrial activity",
                "Check sensor calibration",
                "Document anomaly for trend analysis"
            ])
        else:
            recommendations.extend([
                "Log anomaly for records",
                "Monitor for pattern development"
            ])
        
        # Pollutant-specific recommendations
        if 'PM2.5' in pollutants and pollutants['PM2.5'] > 100:
            recommendations.append("🔍 High PM2.5: Check for vehicular emissions or construction activity")
        
        if 'CO' in pollutants and pollutants['CO'] > 4:
            recommendations.append("🔍 High CO: Investigate traffic congestion or industrial sources")
        
        if 'O3' in pollutants and pollutants['O3'] > 100:
            recommendations.append("🔍 High O3: Likely photochemical smog, advise limiting outdoor activities")
        
        if 'SO2' in pollutants and pollutants['SO2'] > 50:
            recommendations.append("🔍 High SO2: Check industrial emissions, particularly power plants")
        
        return recommendations


class AlertManager:
    """
    Manage collection of alerts.
    """
    
    def __init__(self, output_dir='results'):
        """Initialize alert manager."""
        self.alerts = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = AlertGenerator()
    
    def generate_alerts_from_anomalies(self, df_anomalies, anomaly_scores):
        """
        Generate alerts from detected anomalies.
        
        Args:
            df_anomalies: DataFrame with anomaly data
            anomaly_scores: Array of anomaly scores
            
        Returns:
            list: List of Alert objects
        """
        logger.info(f"Generating alerts for {len(df_anomalies)} anomalies...")
        
        # Calculate 90th percentile for severity classification
        percentile_90 = np.percentile(anomaly_scores, 90)
        
        alerts = []
        for idx, row in df_anomalies.iterrows():
            score = anomaly_scores[idx]
            alert = self.generator.generate_alert(row, score, percentile_90)
            alerts.append(alert)
        
        self.alerts.extend(alerts)
        logger.info(f"Generated {len(alerts)} alerts")
        
        return alerts
    
    def get_active_alerts(self, severity=None):
        """Get all active alerts, optionally filtered by severity."""
        active = [a for a in self.alerts if a.status == 'active']
        
        if severity:
            active = [a for a in active if a.severity == severity]
        
        return active
    
    def get_critical_alerts(self):
        """Get all critical alerts."""
        return self.get_active_alerts(severity='critical')
    
    def get_alerts_by_city(self, city):
        """Get all alerts for a specific city."""
        return [a for a in self.alerts if a.city == city]
    
    def get_alert_summary(self):
        """Get summary statistics of alerts."""
        total = len(self.alerts)
        active = len([a for a in self.alerts if a.status == 'active'])
        
        severity_counts = {
            'critical': len([a for a in self.alerts if a.severity == 'critical']),
            'high': len([a for a in self.alerts if a.severity == 'high']),
            'medium': len([a for a in self.alerts if a.severity == 'medium']),
            'low': len([a for a in self.alerts if a.severity == 'low'])
        }
        
        city_counts = {}
        for alert in self.alerts:
            city_counts[alert.city] = city_counts.get(alert.city, 0) + 1
        
        return {
            'total_alerts': total,
            'active_alerts': active,
            'severity_distribution': severity_counts,
            'alerts_by_city': city_counts
        }
    
    def save_alerts(self, filename='alerts.json'):
        """Save all alerts to file."""
        filepath = self.output_dir / filename
        
        alerts_data = [alert.to_dict() for alert in self.alerts]
        
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=4, default=str)
        
        logger.info(f"Alerts saved to {filepath}")
    
    def export_alerts_csv(self, filename='alerts.csv'):
        """Export alerts to CSV."""
        filepath = self.output_dir / filename
        
        alerts_data = [alert.to_dict() for alert in self.alerts]
        df = pd.DataFrame(alerts_data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Alerts exported to {filepath}")


if __name__ == "__main__":
    # Test alert system
    print("="*80)
    print("Testing Alert System")
    print("="*80)
    
    # Create sample anomaly data
    sample_data = pd.DataFrame({
        'Date': ['2020-11-15', '2020-12-20'],
        'City': ['Delhi', 'Mumbai'],
        'PM2.5': [350, 180],
        'PM10': [450, 220],
        'CO': [5.5, 3.2],
        'NO2': [85, 65],
        'AQI': [450, 280]
    })
    
    # Generate alerts
    manager = AlertManager()
    anomaly_scores = np.array([-0.35, -0.18])
    alerts = manager.generate_alerts_from_anomalies(sample_data, anomaly_scores)
    
    # Display alerts
    for alert in alerts:
        print(f"\n{alert.severity.upper()} ALERT")
        print(f"ID: {alert.alert_id}")
        print(f"City: {alert.city}")
        print(f"Description: {alert.description}")
        print(f"Recommendations:")
        for rec in alert.recommendations:
            print(f"  - {rec}")
    
    # Print summary
    print("\n" + "="*80)
    print("ALERT SUMMARY")
    print("="*80)
    summary = manager.get_alert_summary()
    print(json.dumps(summary, indent=2))
    
    print("\n✅ Alert system working correctly!")
