
export interface SensorData {
  timestamp: string;
  latitude: number;
  longitude: number;
  movement_speed_kmh: number;
  size_estimate: number;
  thermal_signature_c: number;
  vibration_intensity: number;
  object_class: 'Human' | 'Vehicle' | 'Animal' | 'Environmental Noise';
  anomaly_score?: number;
  is_anomaly?: boolean;
  predicted_class?: string;
  alert_priority?: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW/IGNORE';
}

export interface MLMetrics {
  accuracy: number;
  featureImportance: { feature: string; importance: number }[];
}
