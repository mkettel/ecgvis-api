"use client"; // Required for Next.js App Router if using client-side hooks

import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Decimation,
  ChartOptions,
  ChartData
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Decimation
);

// --- TypeScript Interfaces ---
interface TwelveLeadECGDataState {
  time_axis: number[];
  twelve_lead_signals: {
    I: number[];
    II: number[];
    III: number[];
    aVR: number[];
    aVL: number[];
    aVF: number[];
    V1: number[];
    V2: number[];
    V3: number[];
    V4: number[];
    V5: number[];
    V6: number[];
  };
}

interface TwelveLeadECGAPIResponse extends TwelveLeadECGDataState {
  rhythm_generated: string;
}

interface AdvancedRequestBody {
  heart_rate_bpm: number;
  duration_sec: number;
  enable_pvc: boolean;
  pvc_probability_per_sinus: number;
  enable_pac: boolean;
  pac_probability_per_sinus: number;
  first_degree_av_block_pr_sec?: number | null;
  enable_mobitz_ii_av_block?: boolean;
  mobitz_ii_p_waves_per_qrs?: number;
  enable_mobitz_i_wenckebach?: boolean;
  wenckebach_initial_pr_sec?: number;
  wenckebach_pr_increment_sec?: number;
  wenckebach_max_pr_before_drop_sec?: number;
  enable_third_degree_av_block?: boolean;
  third_degree_escape_rhythm_origin?: string;
  third_degree_escape_rate_bpm?: number | null;
  enable_atrial_fibrillation?: boolean;
  afib_average_ventricular_rate_bpm?: number;
  afib_fibrillation_wave_amplitude_mv?: number;
  afib_irregularity_factor?: number;
  enable_atrial_flutter?: boolean;
  atrial_flutter_rate_bpm?: number;
  atrial_flutter_av_block_ratio_qrs_to_f?: number;
  atrial_flutter_wave_amplitude_mv?: number;
  allow_svt_initiation_by_pac?: boolean;
  svt_initiation_probability_after_pac?: number;
  svt_duration_sec?: number;
  svt_rate_bpm?: number;
  enable_vt?: boolean;
  vt_start_time_sec?: number | null;
  vt_duration_sec?: number;
  vt_rate_bpm?: number;
}

// Get environment variables - UPDATE THIS TO YOUR API URL
const API_URL = process.env.NEXT_PUBLIC_EKG_API_URL || 'http://localhost:8000';

// Lead groupings for display
const LIMB_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'];
const PRECORDIAL_LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

// Color scheme for leads
const LEAD_COLORS = {
  I: '#e74c3c',
  II: '#e67e22',
  III: '#f39c12',
  aVR: '#2ecc71',
  aVL: '#27ae60',
  aVF: '#16a085',
  V1: '#3498db',
  V2: '#2980b9',
  V3: '#9b59b6',
  V4: '#8e44ad',
  V5: '#34495e',
  V6: '#2c3e50'
};

const ECGChart12Lead: React.FC = () => {
  const [ecgData, setEcgData] = useState<TwelveLeadECGDataState>({
    time_axis: [],
    twelve_lead_signals: {
      I: [], II: [], III: [], aVR: [], aVL: [], aVF: [],
      V1: [], V2: [], V3: [], V4: [], V5: [], V6: []
    }
  });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [chartTitle, setChartTitle] = useState<string>('12-Lead ECG Simulation');

  // --- Control State (same as single lead) ---
  const [heartRate, setHeartRate] = useState<number>(75);
  const [duration, setDuration] = useState<number>(10);
  const [enablePac, setEnablePac] = useState<boolean>(false);
  const [pacProbability, setPacProbability] = useState<number>(0.1);
  const [enablePvc, setEnablePvc] = useState<boolean>(false);
  const [pvcProbability, setPvcProbability] = useState<number>(0.1);
  const [enableFirstDegreeAVBlock, setEnableFirstDegreeAVBlock] = useState<boolean>(false);
  const [firstDegreePrSec, setFirstDegreePrSec] = useState<number>(0.24);
  const [enableMobitzIIAVBlock, setEnableMobitzIIAVBlock] = useState<boolean>(false);
  const [mobitzIIPWavesPerQRS, setMobitzIIPWavesPerQRS] = useState<number>(3);
  const [enableMobitzIWenckebach, setEnableMobitzIWenckebach] = useState<boolean>(false);
  const [wenckebachInitialPrSec, setWenckebachInitialPrSec] = useState<number>(0.16);
  const [wenckebachPrIncrementSec, setWenckebachPrIncrementSec] = useState<number>(0.04);
  const [wenckebachMaxPrBeforeDropSec, setWenckebachMaxPrBeforeDropSec] = useState<number>(0.32);
  const [enableThirdDegreeAVBlock, setEnableThirdDegreeAVBlock] = useState<boolean>(false);
  const [thirdDegreeEscapeOrigin, setThirdDegreeEscapeOrigin] = useState<string>("junctional");
  const [thirdDegreeEscapeRate, setThirdDegreeEscapeRate] = useState<number>(45);
  const [enableAtrialFibrillation, setEnableAtrialFibrillation] = useState<boolean>(false);
  const [afibVentricularRate, setAfibVentricularRate] = useState<number>(100);
  const [afibAmplitude, setAfibAmplitude] = useState<number>(0.05);
  const [afibIrregularity, setAfibIrregularity] = useState<number>(0.2);
  const [enableAtrialFlutter, setEnableAtrialFlutter] = useState<boolean>(false);
  const [aflutterRate, setAflutterRate] = useState<number>(300);
  const [aflutterConductionRatio, setAflutterConductionRatio] = useState<number>(2);
  const [aflutterAmplitude, setAflutterAmplitude] = useState<number>(0.15);
  const [allowSvtInitiationByPac, setAllowSvtInitiationByPac] = useState<boolean>(false);
  const [svtInitiationProbability, setSvtInitiationProbability] = useState<number>(0.3);
  const [svtDuration, setSvtDuration] = useState<number>(10);
  const [svtRate, setSvtRate] = useState<number>(180);
  const [enableVT, setEnableVT] = useState<boolean>(false);
  const [vtStartTime, setVtStartTime] = useState<number>(0);
  const [vtDuration, setVtDuration] = useState<number>(5);
  const [vtRate, setVtRate] = useState<number>(160);

  const chartRefs = useRef<{ [key: string]: ChartJS<'line', number[], string> | null }>({});

  // Helper booleans for disabling controls
  const isAfibActiveBase = enableAtrialFibrillation;
  const isAflutterActiveBase = enableAtrialFlutter;
  const isThirdDegreeBlockActiveBase = enableThirdDegreeAVBlock;
  const isVTActiveBase = enableVT;
  
  const dominantBaseRhythmOverridesPacSvtOrAVBlocks = isAfibActiveBase || isAflutterActiveBase || isThirdDegreeBlockActiveBase || isVTActiveBase;
  const baseHrDisabled = dominantBaseRhythmOverridesPacSvtOrAVBlocks; 
  const avBlocksDisabled = dominantBaseRhythmOverridesPacSvtOrAVBlocks;
  const pacsAndDynamicSvtSettingsDisabled = dominantBaseRhythmOverridesPacSvtOrAVBlocks;

  const fetchEcgData = async () => {
    setIsLoading(true);
    setError(null);

    // --- Same validations as single lead ---
    if ((enablePac && (pacProbability < 0 || pacProbability > 1)) ||
        (enablePvc && (pvcProbability < 0 || pvcProbability > 1))) {
      setError("Ectopic probabilities must be between 0.0 and 1.0."); setIsLoading(false); return;
    }
    if (enableFirstDegreeAVBlock && (firstDegreePrSec < 0.201 || firstDegreePrSec > 0.60)) {
      setError("1st Degree AV Block PR interval must be between 0.201s and 0.60s."); setIsLoading(false); return;
    }
    if (enableMobitzIIAVBlock && (mobitzIIPWavesPerQRS < 2)) {
      setError("Mobitz II P-waves per QRS must be 2 or greater."); setIsLoading(false); return;
    }
    if (enableMobitzIWenckebach) {
        if (wenckebachInitialPrSec < 0.12 || wenckebachInitialPrSec > 0.40) { setError("Wenckebach Initial PR must be 0.12-0.40s."); setIsLoading(false); return; }
        if (wenckebachPrIncrementSec < 0.01 || wenckebachPrIncrementSec > 0.15) { setError("Wenckebach PR Increment must be 0.01-0.15s."); setIsLoading(false); return; }
        if (wenckebachMaxPrBeforeDropSec < 0.22 || wenckebachMaxPrBeforeDropSec > 0.70 || wenckebachMaxPrBeforeDropSec <= wenckebachInitialPrSec) { setError("Wenckebach Max PR must be 0.22-0.70s & > Initial PR."); setIsLoading(false); return; }
    }
    if (enableThirdDegreeAVBlock && (thirdDegreeEscapeRate < 15 || thirdDegreeEscapeRate > 65)) {
        setError("3rd Degree AV Block Escape Rate must be between 15 and 65 bpm."); setIsLoading(false); return;
    }
    if (enableAtrialFibrillation) {
        if (afibVentricularRate < 30 || afibVentricularRate > 220) {
            setError("Atrial Fibrillation ventricular rate must be between 30 and 220 bpm."); setIsLoading(false); return;
        }
        if (afibAmplitude < 0.0 || afibAmplitude > 0.2) {
            setError("Atrial Fibrillation wave amplitude must be between 0.0 and 0.2mV."); setIsLoading(false); return;
        }
        if (afibIrregularity < 0.05 || afibIrregularity > 0.50) {
            setError("Atrial Fibrillation irregularity factor must be between 0.05 and 0.50."); setIsLoading(false); return;
        }
    }
    if (enableAtrialFlutter) {
        if (aflutterRate < 200 || aflutterRate > 400) {
            setError("Atrial Flutter rate must be between 200 and 400 bpm."); setIsLoading(false); return;
        }
        if (aflutterConductionRatio < 1 ) { 
            setError("Atrial Flutter conduction ratio must be 1 or greater."); setIsLoading(false); return;
        }
        if (aflutterAmplitude < 0.05 || aflutterAmplitude > 0.5) {
          setError("Atrial Flutter wave amplitude must be between 0.05mV and 0.5mV."); setIsLoading(false); return;
        }
    }
    if (allowSvtInitiationByPac) {
        if (svtInitiationProbability < 0.0 || svtInitiationProbability > 1.0) {
            setError("SVT initiation probability must be between 0.0 and 1.0."); setIsLoading(false); return;
        }
        if (svtDuration <= 0 || svtDuration > duration ) {
            setError(`SVT duration must be > 0 and <= total duration (${duration}s).`); setIsLoading(false); return;
        }
        if (svtRate < 150 || svtRate > 250) {
            setError("SVT rate (when active) must be between 150 and 250 bpm."); setIsLoading(false); return;
        }
    }

    // Determine active flags for request body construction
    const sendEnableVT = enableVT;
    const sendEnableAfib = enableAtrialFibrillation && !sendEnableVT;
    const sendEnableAflutter = enableAtrialFlutter && !sendEnableAfib && !sendEnableVT;
    const sendEnableThirdDegreeAVBlock = enableThirdDegreeAVBlock && !sendEnableAfib && !sendEnableAflutter && !sendEnableVT;

    const sendAllowSvtInitiation = allowSvtInitiationByPac && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableVT;
    
    const sendEnableFirstDegreeAVBlock = enableFirstDegreeAVBlock && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableVT;
    const sendEnableMobitzI = enableMobitzIWenckebach && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableVT;
    const sendEnableMobitzII = enableMobitzIIAVBlock && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableMobitzI && !sendEnableVT;
    
    const sendEnablePac = enablePac && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableVT;

    const requestBody: AdvancedRequestBody = {
      heart_rate_bpm: heartRate, 
      duration_sec: duration,
      enable_pac: sendEnablePac, 
      pac_probability_per_sinus: sendEnablePac ? pacProbability : 0,
      enable_pvc: enablePvc, 
      pvc_probability_per_sinus: enablePvc ? pvcProbability : 0,
      
      first_degree_av_block_pr_sec: sendEnableFirstDegreeAVBlock ? firstDegreePrSec : null,
      enable_mobitz_ii_av_block: sendEnableMobitzII,
      mobitz_ii_p_waves_per_qrs: sendEnableMobitzII ? mobitzIIPWavesPerQRS : 2,
      enable_mobitz_i_wenckebach: sendEnableMobitzI,
      wenckebach_initial_pr_sec: sendEnableMobitzI ? wenckebachInitialPrSec : 0.16,
      wenckebach_pr_increment_sec: sendEnableMobitzI ? wenckebachPrIncrementSec : 0.04,
      wenckebach_max_pr_before_drop_sec: sendEnableMobitzI ? wenckebachMaxPrBeforeDropSec : 0.32,
      
      enable_third_degree_av_block: sendEnableThirdDegreeAVBlock,
      third_degree_escape_rhythm_origin: sendEnableThirdDegreeAVBlock ? thirdDegreeEscapeOrigin : "junctional",
      third_degree_escape_rate_bpm: sendEnableThirdDegreeAVBlock ? thirdDegreeEscapeRate : null,
      
      enable_atrial_fibrillation: sendEnableAfib,
      afib_average_ventricular_rate_bpm: sendEnableAfib ? afibVentricularRate : 100,
      afib_fibrillation_wave_amplitude_mv: sendEnableAfib ? afibAmplitude : 0.05,
      afib_irregularity_factor: sendEnableAfib ? afibIrregularity : 0.2,

      enable_atrial_flutter: sendEnableAflutter,
      atrial_flutter_rate_bpm: sendEnableAflutter ? aflutterRate : 300,
      atrial_flutter_av_block_ratio_qrs_to_f: sendEnableAflutter ? aflutterConductionRatio : 2,
      atrial_flutter_wave_amplitude_mv: sendEnableAflutter ? aflutterAmplitude : 0.15,

      allow_svt_initiation_by_pac: sendAllowSvtInitiation,
      svt_initiation_probability_after_pac: sendAllowSvtInitiation ? svtInitiationProbability : 0.3,
      svt_duration_sec: sendAllowSvtInitiation ? svtDuration : 10.0,
      svt_rate_bpm: sendAllowSvtInitiation ? svtRate : 180,
      
      enable_vt: sendEnableVT,
      vt_start_time_sec: sendEnableVT ? vtStartTime : null,
      vt_duration_sec: sendEnableVT ? vtDuration : 5.0,
      vt_rate_bpm: sendEnableVT ? vtRate : 160,
    };

    try {
      const response = await fetch(`${API_URL}/generate_advanced_ecg_12_lead`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', },
        body: JSON.stringify(requestBody),
      });
      if (!response.ok) {
        let errorMessage = `HTTP error! status: ${response.status}`;
        try { const errData = await response.json(); errorMessage = errData.detail || errorMessage; }
        catch (jsonError) { errorMessage = response.statusText || "Unknown server error"; }
        throw new Error(errorMessage);
      }
      const data: TwelveLeadECGAPIResponse = await response.json();
      setEcgData({ 
        time_axis: data.time_axis || [], 
        twelve_lead_signals: data.twelve_lead_signals || {
          I: [], II: [], III: [], aVR: [], aVL: [], aVF: [],
          V1: [], V2: [], V3: [], V4: [], V5: [], V6: []
        }
      });
      setChartTitle(data.rhythm_generated || '12-Lead ECG Simulation');
    } catch (e: any) {
      console.error("Failed to fetch 12-lead ECG data:", e);
      const message = e instanceof Error ? e.message : "An unknown error occurred";
      setError(message);
      setEcgData({ 
        time_axis: [], 
        twelve_lead_signals: {
          I: [], II: [], III: [], aVR: [], aVL: [], aVF: [],
          V1: [], V2: [], V3: [], V4: [], V5: [], V6: []
        }
      });
      setChartTitle('Error generating 12-lead ECG');
    } finally {
      setIsLoading(false);
    }
  };

  // Create enhanced chart data for individual lead
  const createLeadChartData = (leadName: string): ChartData<'line', number[], string> => ({
    labels: ecgData.time_axis.map(t => t.toFixed(2)),
    datasets: [
      {
        label: `Lead ${leadName}`,
        data: ecgData.twelve_lead_signals[leadName as keyof typeof ecgData.twelve_lead_signals] || [],
        borderColor: LEAD_COLORS[leadName as keyof typeof LEAD_COLORS],
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
      },
    ],
  });

  // Enhanced chart options for individual leads with ECG styling
  const getLeadChartOptions = (leadName: string): ChartOptions<'line'> => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: {
      intersect: false,
      mode: 'index',
    },
    scales: {
      x: {
        title: { 
          display: false 
        },
        ticks: {
          maxTicksLimit: 12,
          autoSkipPadding: 5,
          font: { size: 9 },
          color: '#999999',
        },
        grid: { 
          color: 'rgba(255, 182, 193, 0.4)', // ECG paper pink grid
          lineWidth: 0.5
        }
      },
      y: {
        title: { 
          display: false
        },
        min: -1.5,
        max: 1.5,
        ticks: { 
          font: { size: 9 },
          color: '#999999',
          stepSize: 0.5,
          callback: function(value) {
            return `${value}mV`;
          }
        },
        grid: { 
          color: 'rgba(255, 182, 193, 0.4)', // ECG paper pink grid
          lineWidth: 0.5
        }
      },
    },
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: leadName,
        color: LEAD_COLORS[leadName as keyof typeof LEAD_COLORS],
        font: { size: 16, weight: 'bold' },
        padding: { top: 8, bottom: 12 }
      },
      tooltip: {
        enabled: false
      },
      decimation: {
        enabled: true,
        algorithm: 'lttb',
        samples: Math.min(1000, ecgData.time_axis.length || 500),
      },
    },
  });

  // Event handlers (same as single lead but simplified for brevity)
  const handleHeartRateChange = (e: React.ChangeEvent<HTMLInputElement>) => setHeartRate(parseFloat(e.target.value) || 0);
  const handleDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => setDuration(parseFloat(e.target.value) || 0);
  const handleEnablePacChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnablePac(isEnabled);
    if (!isEnabled) {
        setAllowSvtInitiationByPac(false);
    }
  };
  const handlePacProbabilityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setPacProbability(isNaN(val) ? 0 : Math.max(0, Math.min(1, val)));
  };
  const handleEnablePvcChange = (e: React.ChangeEvent<HTMLInputElement>) => setEnablePvc(e.target.checked);
  const handlePvcProbabilityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setPvcProbability(isNaN(val) ? 0 : Math.max(0, Math.min(1, val)));
  };
  
  // Style functions
  const toggleStyles = (isDisabled: boolean) => 
    `block h-5 w-10 cursor-pointer rounded-full ${isDisabled ? 'cursor-not-allowed bg-gray-300' : 'peer-checked:after:translate-x-full peer-checked:bg-blue-600'} after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:bg-white after:transition-all bg-gray-200`;
    
  const rangeSliderStyles = (isDisabled: boolean) =>
    `h-1 w-full cursor-pointer appearance-none rounded-lg ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''} bg-gray-200 accent-blue-600`;
    
  const numberInputStyles = (isDisabled: boolean) =>
    `w-full border rounded-md px-2 py-1.5 text-sm ${isDisabled ? 'opacity-50 cursor-not-allowed' : 'focus:ring-blue-500 focus:border-blue-500'} border-gray-300 bg-white text-gray-900`;
    
  const labelTextStyles = (isActive: boolean, isDisabled?: boolean) => {
    if (isDisabled) return 'text-sm font-medium text-gray-400';
    return `text-sm font-medium ${isActive ? 'text-gray-900' : 'text-gray-600'}`;
  };

  useEffect(() => { fetchEcgData(); }, []);

  return (
    <div className="overflow-auto h-screen flex flex-col bg-gray-50">
      <div className="px-4 py-4 mx-auto w-full max-w-8xl">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          {/* Header */}
          <div className="lg:col-span-5 mb-4 flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">12-Lead ECG Simulator</h1>
              <p className="text-gray-600">Generate vectorized 12-lead ECG with physiological accuracy</p>
            </div>
          </div>
          
          {/* Controls Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl p-6 shadow-lg h-full overflow-y-auto max-h-[calc(100vh-150px)]">
              {/* Basic Controls */}
              <div className="mb-6 pb-5 border-b border-gray-200">
                <h2 className="text-lg font-semibold mb-3 text-gray-900">Basic Settings</h2>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className={`text-sm font-medium ${baseHrDisabled ? 'text-gray-400' : 'text-gray-700'}`}>
                        Heart Rate (bpm)
                      </label>
                      <span className={`text-lg font-medium ${baseHrDisabled ? 'text-gray-400' : 'text-gray-900'}`}>{heartRate}</span>
                    </div>
                    <input type="range" value={heartRate} onChange={handleHeartRateChange} min="30" max="250" 
                           disabled={baseHrDisabled} className={rangeSliderStyles(baseHrDisabled)}/>
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-700">Duration (seconds)</label>
                      <span className="text-lg font-medium text-gray-900">{duration}</span>
                    </div>
                    <input type="range" value={duration} onChange={handleDurationChange} min="1" max="60" className={rangeSliderStyles(false)}/>
                  </div>
                </div>
              </div>

              {/* Arrhythmias */}
              <div className="mb-6 pb-5 border-b border-gray-200">
                <h2 className="text-lg font-semibold mb-3 text-gray-900">Arrhythmias</h2>
                <div className="space-y-4">
                  
                  {/* Atrial Fibrillation */}
                  <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className={labelTextStyles(enableAtrialFibrillation)}>Atrial Fibrillation</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableAfibCheckbox" checked={enableAtrialFibrillation} onChange={(e) => {
                          const isEnabled = e.target.checked;
                          setEnableAtrialFibrillation(isEnabled);
                          if (isEnabled) {
                            setEnableAtrialFlutter(false);
                            setEnableThirdDegreeAVBlock(false);
                            setEnableVT(false);
                          }
                        }} className="sr-only peer"/>
                        <label htmlFor="enableAfibCheckbox" className={toggleStyles(false)}></label>
                      </div>
                    </div>
                    {enableAtrialFibrillation && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className="text-xs text-gray-600">Ventricular Rate</label>
                            <span className="text-xs text-gray-800">{afibVentricularRate} bpm</span>
                          </div>
                          <input type="range" value={afibVentricularRate} onChange={(e) => {
                            const val = parseInt(e.target.value, 10);
                            setAfibVentricularRate(isNaN(val) ? 80 : Math.max(40, Math.min(200, val)));
                          }} min="40" max="200" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className="text-xs text-gray-600">Irregularity</label>
                            <span className="text-xs text-gray-800">{afibIrregularity.toFixed(2)}</span>
                          </div>
                          <input type="range" value={afibIrregularity} onChange={(e) => {
                            const val = parseFloat(e.target.value);
                            setAfibIrregularity(isNaN(val) ? 0.2 : Math.max(0.1, Math.min(0.5, val)));
                          }} min="0.1" max="0.5" step="0.1" className={rangeSliderStyles(false)}/>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Atrial Flutter */}
                  <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className={labelTextStyles(enableAtrialFlutter, enableAtrialFibrillation || enableThirdDegreeAVBlock || enableVT)}>Atrial Flutter</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableAflutterCheckbox" checked={enableAtrialFlutter} onChange={(e) => {
                          const isEnabled = e.target.checked;
                          setEnableAtrialFlutter(isEnabled);
                          if (isEnabled) {
                            setEnableAtrialFibrillation(false);
                            setEnableThirdDegreeAVBlock(false);
                            setEnableVT(false);
                          }
                        }} 
                               disabled={enableAtrialFibrillation || enableThirdDegreeAVBlock || enableVT} className="sr-only peer"/>
                        <label htmlFor="enableAflutterCheckbox" className={toggleStyles(enableAtrialFibrillation || enableThirdDegreeAVBlock || enableVT)}></label>
                      </div>
                    </div>
                    {enableAtrialFlutter && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className="text-xs text-gray-600">Atrial Rate</label>
                            <span className="text-xs text-gray-800">{aflutterRate} bpm</span>
                          </div>
                          <input type="range" value={aflutterRate} onChange={(e) => {
                            const val = parseInt(e.target.value, 10);
                            setAflutterRate(isNaN(val) ? 300 : Math.max(240, Math.min(400, val)));
                          }} min="240" max="400" step="10" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className="text-xs text-gray-600">AV Block Ratio</label>
                            <span className="text-xs text-gray-800">{aflutterConductionRatio}:1</span>
                          </div>
                          <input type="range" value={aflutterConductionRatio} onChange={(e) => {
                            const val = parseInt(e.target.value, 10);
                            setAflutterConductionRatio(isNaN(val) ? 2 : Math.max(2, Math.min(6, val)));
                          }} min="2" max="6" className={rangeSliderStyles(false)}/>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Ventricular Tachycardia */}
                  <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className={labelTextStyles(enableVT, enableAtrialFibrillation || enableAtrialFlutter || enableThirdDegreeAVBlock)}>Ventricular Tachycardia</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableVTCheckbox" checked={enableVT} onChange={(e) => {
                          const isEnabled = e.target.checked;
                          setEnableVT(isEnabled);
                          if (isEnabled) {
                            setEnableAtrialFibrillation(false);
                            setEnableAtrialFlutter(false);
                            setEnableThirdDegreeAVBlock(false);
                          }
                        }} 
                               disabled={enableAtrialFibrillation || enableAtrialFlutter || enableThirdDegreeAVBlock} className="sr-only peer"/>
                        <label htmlFor="enableVTCheckbox" className={toggleStyles(enableAtrialFibrillation || enableAtrialFlutter || enableThirdDegreeAVBlock)}></label>
                      </div>
                    </div>
                    {enableVT && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className="text-xs text-gray-600">Start Time</label>
                            <span className="text-xs text-gray-800">{vtStartTime}s</span>
                          </div>
                          <input type="range" value={vtStartTime} onChange={(e) => {
                            const val = parseFloat(e.target.value);
                            setVtStartTime(isNaN(val) ? 0 : Math.max(0, Math.min(duration-2, val)));
                          }} min="0" max={duration-2} step="0.5" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className="text-xs text-gray-600">Duration</label>
                            <span className="text-xs text-gray-800">{vtDuration}s</span>
                          </div>
                          <input type="range" value={vtDuration} onChange={(e) => {
                            const val = parseFloat(e.target.value);
                            setVtDuration(isNaN(val) ? 1 : Math.max(1, Math.min(duration-vtStartTime, val)));
                          }} min="1" max={duration-vtStartTime} step="0.5" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className="text-xs text-gray-600">VT Rate</label>
                            <span className="text-xs text-gray-800">{vtRate} bpm</span>
                          </div>
                          <input type="range" value={vtRate} onChange={(e) => {
                            const val = parseInt(e.target.value, 10);
                            setVtRate(isNaN(val) ? 150 : Math.max(120, Math.min(250, val)));
                          }} min="120" max="250" step="10" className={rangeSliderStyles(false)}/>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Ectopy Controls */}
              <div className="mb-6 pb-5 border-b border-gray-200">
                <h2 className="text-lg font-semibold mb-3 text-gray-900">Ectopy</h2>
                <div className="space-y-4">
                  {/* PACs */}
                  <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className={labelTextStyles(enablePac, pacsAndDynamicSvtSettingsDisabled)}>PACs</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enablePacCheckbox" checked={enablePac} onChange={handleEnablePacChange} 
                               disabled={pacsAndDynamicSvtSettingsDisabled} className="sr-only peer"/>
                        <label htmlFor="enablePacCheckbox" className={toggleStyles(pacsAndDynamicSvtSettingsDisabled)}></label>
                      </div>
                    </div>
                    {enablePac && !pacsAndDynamicSvtSettingsDisabled && (
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <label className="text-xs text-gray-600">Probability</label>
                          <span className="text-xs text-gray-800">{pacProbability.toFixed(2)}</span>
                        </div>
                        <input type="range" value={pacProbability} onChange={handlePacProbabilityChange} min="0" max="0.3" step="0.01" className={rangeSliderStyles(false)}/>
                      </div>
                    )}
                  </div>

                  {/* PVCs */}
                  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className={labelTextStyles(enablePvc)}>PVCs</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enablePvcCheckbox" checked={enablePvc} onChange={handleEnablePvcChange} className="sr-only peer"/>
                        <label htmlFor="enablePvcCheckbox" className={toggleStyles(false)}></label>
                      </div>
                    </div>
                    {enablePvc && (
                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <label className="text-xs text-gray-600">Probability</label>
                          <span className="text-xs text-gray-800">{pvcProbability.toFixed(2)}</span>
                        </div>
                        <input type="range" value={pvcProbability} onChange={handlePvcProbabilityChange} min="0" max="0.3" step="0.01" className={rangeSliderStyles(false)}/>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Generate Button */}
              <div className="sticky bottom-0 left-0 right-0 pt-4">
                <button 
                  onClick={fetchEcgData} 
                  disabled={isLoading} 
                  className={`w-full px-4 py-3 rounded-lg text-white font-medium shadow transition-all ${
                    isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
                  }`}
                >
                  {isLoading ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Generating...
                    </span>
                  ) : (
                    'Generate 12-Lead ECG'
                  )}
                </button>
              </div>
            </div>
          </div>
          
          {/* 12-Lead Display */}
          <div className="lg:col-span-4">
            <div className="bg-white rounded-xl overflow-hidden shadow-lg h-[calc(100vh-150px)] flex flex-col">
              <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                <div className="font-medium text-gray-900">{chartTitle}</div>
                <div className="text-sm text-gray-600">
                  {heartRate} bpm, {duration}s
                </div>
              </div>

              {error && (
                <div className="m-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md text-sm flex items-center">
                  <svg className="mr-2 flex-shrink-0" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                  <span>{error}</span>
                </div>
              )}

              <div className="p-4 flex-grow relative">
                {isLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-50 bg-opacity-75 z-10">
                    <div className="text-center">
                      <div className="animate-pulse flex space-x-2 justify-center mb-2">
                        <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                      </div>
                      <p className="text-sm text-gray-600">Generating 12-lead ECG...</p>
                    </div>
                  </div>
                )}

                {!isLoading && ecgData.time_axis.length > 0 && (
                  <div className="h-full overflow-auto">
                    {/* Limb Leads */}
                    <div className="mb-8">
                      <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                        <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                        Limb Leads
                      </h3>
                      <div className="grid grid-cols-3 gap-4">
                        {LIMB_LEADS.map((lead) => (
                          <div key={lead} className="h-48 bg-gradient-to-br from-gray-50 to-white rounded-lg border border-gray-200 p-3 shadow-sm">
                            <Line
                              ref={(ref) => { chartRefs.current[lead] = ref; }}
                              options={getLeadChartOptions(lead)}
                              data={createLeadChartData(lead)}
                            />
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Precordial Leads */}
                    <div>
                      <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                        <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                        Precordial Leads
                      </h3>
                      <div className="grid grid-cols-3 gap-4">
                        {PRECORDIAL_LEADS.map((lead) => (
                          <div key={lead} className="h-48 bg-gradient-to-br from-gray-50 to-white rounded-lg border border-gray-200 p-3 shadow-sm">
                            <Line
                              ref={(ref) => { chartRefs.current[lead] = ref; }}
                              options={getLeadChartOptions(lead)}
                              data={createLeadChartData(lead)}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {!isLoading && ecgData.time_axis.length === 0 && !error && (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <svg className="mx-auto h-12 w-12 mb-4 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <p className="text-gray-500">No ECG data to display</p>
                      <p className="text-sm text-gray-400 mt-1">Click "Generate 12-Lead ECG" to start</p>
                    </div>
                  </div>
                )}
              </div>

              <div className="px-4 py-2 border-t border-gray-200 text-xs text-gray-500">
                * This is a simulated 12-lead ECG for educational purposes only
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ECGChart12Lead;