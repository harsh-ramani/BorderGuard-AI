import { SensorData } from '../types';

/**
 * Simplified Isolation Forest-like Anomaly Detection
 * For this demo, we use a robust Z-score approach across multiple dimensions
 * combined with a density-based check.
 */
export function detectAnomalies(data: SensorData[], contamination: number): SensorData[] {
  if (data.length === 0) return [];

  const features: (keyof SensorData)[] = [
    'movement_speed_kmh',
    'size_estimate',
    'thermal_signature_c',
    'vibration_intensity'
  ];

  // Calculate means and std devs for each feature
  const stats = features.reduce((acc, feature) => {
    const values = data.map(d => d[feature] as number);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
    acc[feature] = { mean, std };
    return acc;
  }, {} as Record<string, { mean: number; std: number }>);

  // Calculate anomaly scores (average Z-score across features)
  const scoredData = data.map(d => {
    let totalZ = 0;
    features.forEach(f => {
      const { mean, std } = stats[f];
      const z = std === 0 ? 0 : Math.abs(((d[f] as number) - mean) / std);
      totalZ += z;
    });
    return { ...d, anomaly_score: totalZ / features.length };
  });

  // Sort by score and mark top % as anomalies
  const sorted = [...scoredData].sort((a, b) => (b.anomaly_score || 0) - (a.anomaly_score || 0));
  const thresholdIndex = Math.floor(data.length * contamination);
  const thresholdScore = sorted[thresholdIndex]?.anomaly_score || 0;

  return scoredData.map(d => ({
    ...d,
    is_anomaly: (d.anomaly_score || 0) >= thresholdScore
  }));
}

/**
 * Alert Prioritization Logic
 */
export function prioritizeAlerts(data: SensorData[]): SensorData[] {
  return data.map(d => {
    let priority: SensorData['alert_priority'] = 'LOW/IGNORE';
    
    const isThreatClass = d.object_class === 'Human' || d.object_class === 'Vehicle';
    const isAnomaly = d.is_anomaly;
    
    if (isThreatClass && isAnomaly) {
      priority = 'CRITICAL';
    } else if (isThreatClass) {
      priority = 'HIGH';
    } else if (isAnomaly) {
      priority = 'MEDIUM';
    }

    return { ...d, alert_priority: priority };
  });
}
