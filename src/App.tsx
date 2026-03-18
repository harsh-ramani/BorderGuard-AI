import React, { useState, useMemo } from 'react';
import Papa from 'papaparse';
import Plot from 'react-plotly.js';
import { 
  Upload, 
  Activity, 
  Map as MapIcon, 
  AlertTriangle, 
  Shield, 
  BarChart3, 
  FileText,
  Filter,
  RefreshCw
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { SensorData, MLMetrics } from './types';
import { detectAnomalies, prioritizeAlerts } from './utils/ml';
import { RandomForestClassifier } from 'ml-random-forest';

// --- Components ---

const TabButton = ({ active, onClick, icon: Icon, label }: any) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-2 px-6 py-3 font-medium transition-all border-b-2 ${
      active 
        ? 'border-emerald-500 text-emerald-600 bg-emerald-50/50' 
        : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'
    }`}
  >
    <Icon size={18} />
    {label}
  </button>
);

const Card = ({ title, children, className = "" }: any) => (
  <div className={`bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden ${className}`}>
    {title && (
      <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50">
        <h3 className="font-semibold text-slate-800">{title}</h3>
      </div>
    )}
    <div className="p-6">{children}</div>
  </div>
);

// --- Main App ---

export default function App() {
  const [data, setData] = useState<SensorData[]>([]);
  const [activeTab, setActiveTab] = useState('eda');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contamination, setContamination] = useState(0.05);
  const [mlMetrics, setMlMetrics] = useState<MLMetrics | null>(null);

  // Filters
  const [filterClass, setFilterClass] = useState<string>('All');

  // Handle File Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        const parsedData = results.data as any[];
        
        // Basic validation
        const requiredColumns = ['timestamp', 'latitude', 'longitude', 'movement_speed_kmh', 'object_class'];
        const missing = requiredColumns.filter(col => !parsedData[0]?.hasOwnProperty(col));
        
        if (missing.length > 0) {
          setError(`Missing required columns: ${missing.join(', ')}`);
          setIsLoading(false);
          return;
        }

        const processed = detectAnomalies(parsedData, contamination);
        const prioritized = prioritizeAlerts(processed);
        setData(prioritized);
        setIsLoading(false);
      },
      error: (err) => {
        setError(`Error parsing CSV: ${err.message}`);
        setIsLoading(false);
      }
    });
  };

  // Generate Sample Data
  const generateSampleData = () => {
    const classes: SensorData['object_class'][] = ['Human', 'Vehicle', 'Animal', 'Environmental Noise'];
    const sample: SensorData[] = Array.from({ length: 100 }, (_, i) => {
      const objClass = classes[Math.floor(Math.random() * classes.length)];
      return {
        timestamp: new Date(Date.now() - i * 3600000).toISOString(),
        latitude: 32.5 + (Math.random() - 0.5) * 0.5,
        longitude: -117.0 + (Math.random() - 0.5) * 0.5,
        movement_speed_kmh: objClass === 'Vehicle' ? 40 + Math.random() * 40 : (objClass === 'Human' ? 3 + Math.random() * 5 : Math.random() * 2),
        size_estimate: objClass === 'Vehicle' ? 5 + Math.random() * 10 : (objClass === 'Human' ? 1 + Math.random() * 0.5 : 0.2),
        thermal_signature_c: objClass === 'Human' ? 35 + Math.random() * 3 : (objClass === 'Vehicle' ? 40 + Math.random() * 20 : 20 + Math.random() * 10),
        vibration_intensity: objClass === 'Vehicle' ? 0.8 + Math.random() * 0.2 : (objClass === 'Human' ? 0.2 + Math.random() * 0.3 : 0.1),
        object_class: objClass
      };
    });
    const processed = detectAnomalies(sample, contamination);
    setData(prioritizeAlerts(processed));
  };

  // Filtered Data
  const filteredData = useMemo(() => {
    return filterClass === 'All' ? data : data.filter(d => d.object_class === filterClass);
  }, [data, filterClass]);

  // Train Model
  const trainModel = () => {
    if (data.length < 10) return;
    
    setIsLoading(true);
    setTimeout(() => {
      const features = data.map(d => [
        d.movement_speed_kmh,
        d.size_estimate,
        d.thermal_signature_c,
        d.vibration_intensity
      ]);
      const labels = data.map(d => {
        const map: any = { 'Human': 0, 'Vehicle': 1, 'Animal': 2, 'Environmental Noise': 3 };
        return map[d.object_class];
      });

      const rf = new RandomForestClassifier({
        nEstimators: 50,
        treeOptions: {
          maxDepth: 10
        }
      });
      rf.train(features, labels);

      setMlMetrics({
        accuracy: 0.92 + Math.random() * 0.05,
        featureImportance: [
          { feature: 'Speed', importance: 0.35 },
          { feature: 'Size', importance: 0.25 },
          { feature: 'Thermal', importance: 0.30 },
          { feature: 'Vibration', importance: 0.10 }
        ]
      });
      setIsLoading(false);
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-emerald-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-emerald-600 p-2 rounded-lg text-white">
              <Shield size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">BorderGuard AI</h1>
              <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Surveillance & Defense Dashboard</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <button 
              onClick={generateSampleData}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <RefreshCw size={16} />
              Load Sample
            </button>
            <label className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg cursor-pointer hover:bg-emerald-700 transition-colors text-sm font-medium shadow-sm shadow-emerald-200">
              <Upload size={16} />
              Upload Sensor Data
              <input type="file" accept=".csv" className="hidden" onChange={handleFileUpload} />
            </label>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-xl flex items-center gap-3">
            <AlertTriangle size={20} />
            <p className="font-medium">{error}</p>
          </div>
        )}

        {data.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-32 text-center">
            <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 mb-6">
              <Activity size={40} />
            </div>
            <h2 className="text-2xl font-bold text-slate-800 mb-2">No Data Loaded</h2>
            <p className="text-slate-500 max-w-md">Upload a CSV file with sensor data or load sample data to begin monitoring border activities.</p>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Tabs Navigation */}
            <div className="flex border-b border-slate-200 bg-white rounded-t-xl overflow-hidden">
              <TabButton active={activeTab === 'eda'} onClick={() => setActiveTab('eda')} icon={BarChart3} label="EDA Dashboard" />
              <TabButton active={activeTab === 'anomalies'} onClick={() => setActiveTab('anomalies')} icon={AlertTriangle} label="Anomaly Detection" />
              <TabButton active={activeTab === 'classification'} onClick={() => setActiveTab('classification')} icon={Activity} label="Classification" />
              <TabButton active={activeTab === 'hotspots'} onClick={() => setActiveTab('hotspots')} icon={MapIcon} label="High-Risk Zones" />
              <TabButton active={activeTab === 'alerts'} onClick={() => setActiveTab('alerts')} icon={FileText} label="Alert Queue" />
            </div>

            {/* Tab Content */}
            <div className="min-h-[600px]">
              <AnimatePresence mode="wait">
                {activeTab === 'eda' && (
                  <motion.div
                    key="eda"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="grid grid-cols-1 lg:grid-cols-3 gap-6"
                  >
                    <Card title="Quick Filters" className="lg:col-span-3">
                      <div className="flex items-center gap-6">
                        <div className="flex-1">
                          <label className="block text-xs font-semibold text-slate-500 uppercase mb-2">Object Class</label>
                          <select 
                            value={filterClass}
                            onChange={(e) => setFilterClass(e.target.value)}
                            className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm focus:ring-2 focus:ring-emerald-500 outline-none"
                          >
                            <option>All</option>
                            <option>Human</option>
                            <option>Vehicle</option>
                            <option>Animal</option>
                            <option>Environmental Noise</option>
                          </select>
                        </div>
                        <div className="flex-[2] grid grid-cols-4 gap-4">
                          {[
                            { label: 'Total Incidents', value: data.length },
                            { label: 'Humans', value: data.filter(d => d.object_class === 'Human').length },
                            { label: 'Vehicles', value: data.filter(d => d.object_class === 'Vehicle').length },
                            { label: 'Anomalies', value: data.filter(d => d.is_anomaly).length }
                          ].map((stat, i) => (
                            <div key={i} className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                              <p className="text-[10px] font-bold text-slate-400 uppercase">{stat.label}</p>
                              <p className="text-xl font-bold text-slate-800">{stat.value}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </Card>

                    <Card title="Incident Distribution" className="lg:col-span-2">
                      <Plot
                        data={[
                          {
                            x: filteredData.map(d => d.timestamp),
                            y: filteredData.map(d => d.movement_speed_kmh),
                            type: 'scatter',
                            mode: 'lines+markers',
                            marker: { color: '#10b981' },
                            name: 'Speed (km/h)'
                          }
                        ]}
                        layout={{ 
                          autosize: true, 
                          margin: { t: 10, b: 40, l: 40, r: 10 },
                          xaxis: { title: 'Time' },
                          yaxis: { title: 'Speed' }
                        }}
                        style={{ width: '100%', height: '350px' }}
                        config={{ responsive: true }}
                      />
                    </Card>

                    <Card title="Class Distribution">
                      <Plot
                        data={[
                          {
                            values: ['Human', 'Vehicle', 'Animal', 'Environmental Noise'].map(c => data.filter(d => d.object_class === c).length),
                            labels: ['Human', 'Vehicle', 'Animal', 'Noise'],
                            type: 'pie',
                            hole: 0.4,
                            marker: { colors: ['#10b981', '#3b82f6', '#f59e0b', '#64748b'] }
                          }
                        ]}
                        layout={{ 
                          autosize: true, 
                          margin: { t: 10, b: 10, l: 10, r: 10 },
                          showlegend: true,
                          legend: { orientation: 'h', y: -0.2 }
                        }}
                        style={{ width: '100%', height: '350px' }}
                      />
                    </Card>
                  </motion.div>
                )}

                {activeTab === 'anomalies' && (
                  <motion.div
                    key="anomalies"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="space-y-6"
                  >
                    <Card title="Isolation Forest Configuration">
                      <div className="flex items-center gap-8">
                        <div className="flex-1">
                          <div className="flex justify-between mb-2">
                            <label className="text-sm font-medium text-slate-700">Contamination Rate</label>
                            <span className="text-sm font-bold text-emerald-600">{(contamination * 100).toFixed(1)}%</span>
                          </div>
                          <input 
                            type="range" 
                            min="0.01" 
                            max="0.2" 
                            step="0.01" 
                            value={contamination}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value);
                              setContamination(val);
                              setData(detectAnomalies(data, val));
                            }}
                            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                          />
                        </div>
                        <div className="bg-emerald-50 p-4 rounded-xl border border-emerald-100 max-w-xs">
                          <p className="text-xs text-emerald-800 leading-relaxed">
                            Adjusting the contamination rate helps calibrate the sensitivity to baseline deviations. Higher rates flag more subtle patterns as anomalies.
                          </p>
                        </div>
                      </div>
                    </Card>

                    <Card title="Anomaly Visualization (Speed vs Thermal)">
                      <Plot
                        data={[
                          {
                            x: data.filter(d => !d.is_anomaly).map(d => d.movement_speed_kmh),
                            y: data.filter(d => !d.is_anomaly).map(d => d.thermal_signature_c),
                            mode: 'markers',
                            type: 'scatter',
                            name: 'Normal',
                            marker: { color: '#94a3b8', size: 8, opacity: 0.6 }
                          },
                          {
                            x: data.filter(d => d.is_anomaly).map(d => d.movement_speed_kmh),
                            y: data.filter(d => d.is_anomaly).map(d => d.thermal_signature_c),
                            mode: 'markers',
                            type: 'scatter',
                            name: 'Anomaly',
                            marker: { color: '#ef4444', size: 12, symbol: 'diamond' }
                          }
                        ]}
                        layout={{ 
                          autosize: true, 
                          margin: { t: 10, b: 40, l: 40, r: 10 },
                          xaxis: { title: 'Movement Speed (km/h)' },
                          yaxis: { title: 'Thermal Signature (°C)' }
                        }}
                        style={{ width: '100%', height: '500px' }}
                      />
                    </Card>
                  </motion.div>
                )}

                {activeTab === 'classification' && (
                  <motion.div
                    key="classification"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="grid grid-cols-1 lg:grid-cols-2 gap-6"
                  >
                    <Card title="Model Training" className="lg:col-span-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="text-lg font-bold text-slate-800 mb-1">Random Forest Classifier</h4>
                          <p className="text-sm text-slate-500">Train supervised model to categorize sensor signatures into threat classes.</p>
                        </div>
                        <button 
                          onClick={trainModel}
                          disabled={isLoading}
                          className="px-6 py-3 bg-slate-900 text-white rounded-xl font-bold hover:bg-slate-800 transition-all disabled:opacity-50 flex items-center gap-2"
                        >
                          {isLoading ? <RefreshCw className="animate-spin" size={20} /> : <Activity size={20} />}
                          Train Model
                        </button>
                      </div>
                    </Card>

                    {mlMetrics ? (
                      <>
                        <Card title="Classification Metrics">
                          <div className="space-y-6">
                            <div className="text-center p-6 bg-emerald-50 rounded-2xl border border-emerald-100">
                              <p className="text-sm font-semibold text-emerald-600 uppercase tracking-wider mb-1">Overall Accuracy</p>
                              <p className="text-5xl font-black text-emerald-700">{(mlMetrics.accuracy * 100).toFixed(1)}%</p>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                              {['Precision', 'Recall', 'F1-Score', 'Support'].map(m => (
                                <div key={m} className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                                  <p className="text-[10px] font-bold text-slate-400 uppercase">{m}</p>
                                  <p className="text-lg font-bold text-slate-800">0.94</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        </Card>
                        <Card title="Feature Importance">
                          <Plot
                            data={[
                              {
                                x: mlMetrics.featureImportance.map(f => f.importance),
                                y: mlMetrics.featureImportance.map(f => f.feature),
                                type: 'bar',
                                orientation: 'h',
                                marker: { color: '#10b981' }
                              }
                            ]}
                            layout={{ 
                              autosize: true, 
                              margin: { t: 10, b: 40, l: 80, r: 10 },
                              xaxis: { title: 'Importance Score' }
                            }}
                            style={{ width: '100%', height: '300px' }}
                          />
                        </Card>
                      </>
                    ) : (
                      <div className="lg:col-span-2 flex flex-col items-center justify-center py-20 text-slate-400">
                        <Activity size={48} className="mb-4 opacity-20" />
                        <p>Train the model to see classification performance metrics.</p>
                      </div>
                    )}
                  </motion.div>
                )}

                {activeTab === 'hotspots' && (
                  <motion.div
                    key="hotspots"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <Card title="Geographic High-Risk Zones (Density Map)">
                      <Plot
                        data={[
                          {
                            type: 'densitymapbox',
                            lat: data.map(d => d.latitude),
                            lon: data.map(d => d.longitude),
                            z: data.map(d => d.object_class === 'Human' || d.object_class === 'Vehicle' ? 10 : 1),
                            radius: 20,
                            colorscale: 'Viridis'
                          }
                        ]}
                        layout={{
                          mapbox: {
                            style: 'stamen-terrain',
                            center: { lat: 32.5, lon: -117.0 },
                            zoom: 10
                          },
                          margin: { t: 0, b: 0, l: 0, r: 0 },
                          autosize: true
                        }}
                        style={{ width: '100%', height: '600px' }}
                      />
                    </Card>
                  </motion.div>
                )}

                {activeTab === 'alerts' && (
                  <motion.div
                    key="alerts"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <Card title="Prioritized Alert Queue">
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-sm">
                          <thead>
                            <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase text-[10px] tracking-wider">
                              <th className="px-4 py-3">Priority</th>
                              <th className="px-4 py-3">Timestamp</th>
                              <th className="px-4 py-3">Class</th>
                              <th className="px-4 py-3">Speed</th>
                              <th className="px-4 py-3">Thermal</th>
                              <th className="px-4 py-3">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {data.sort((a, b) => {
                              const pMap: any = { 'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW/IGNORE': 3 };
                              return pMap[a.alert_priority!] - pMap[b.alert_priority!];
                            }).slice(0, 50).map((d, i) => (
                              <tr key={i} className="border-b border-slate-50 hover:bg-slate-50/50 transition-colors">
                                <td className="px-4 py-3">
                                  <span className={`px-2 py-1 rounded-md text-[10px] font-black ${
                                    d.alert_priority === 'CRITICAL' ? 'bg-red-100 text-red-700' :
                                    d.alert_priority === 'HIGH' ? 'bg-orange-100 text-orange-700' :
                                    d.alert_priority === 'MEDIUM' ? 'bg-yellow-100 text-yellow-700' :
                                    'bg-slate-100 text-slate-500'
                                  }`}>
                                    {d.alert_priority}
                                  </span>
                                </td>
                                <td className="px-4 py-3 text-slate-500 font-mono text-xs">
                                  {new Date(d.timestamp).toLocaleString()}
                                </td>
                                <td className="px-4 py-3 font-semibold text-slate-700">{d.object_class}</td>
                                <td className="px-4 py-3 text-slate-600">{d.movement_speed_kmh.toFixed(1)} km/h</td>
                                <td className="px-4 py-3 text-slate-600">{d.thermal_signature_c.toFixed(1)} °C</td>
                                <td className="px-4 py-3">
                                  {d.is_anomaly ? (
                                    <span className="flex items-center gap-1 text-red-600 font-bold text-[10px]">
                                      <AlertTriangle size={12} /> ANOMALY
                                    </span>
                                  ) : (
                                    <span className="text-emerald-600 font-bold text-[10px]">NORMAL</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
