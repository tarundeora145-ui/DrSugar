import { useState, useRef } from 'react';
import { ExplainabilityPanel } from '../components/ExplainabilityPanel';
import { Link } from 'react-router-dom';

type Step = 'PATIENT' | 'UPLOAD' | 'QUALITY' | 'ENHANCEMENT' | 'ANALYSIS' | 'ASSESSMENT' | 'RESULTS';

const PIPELINE_STEPS: Step[] = [
  'PATIENT', 'UPLOAD', 'QUALITY', 'ENHANCEMENT', 'ANALYSIS', 'ASSESSMENT', 'RESULTS'
];

const API_BASE_URL = '/api';

const Screening = () => {
  const [currentStep, setCurrentStep] = useState<Step>('PATIENT');
  const [status, setStatus] = useState<'IDLE' | 'PROCESSING' | 'ERROR' | 'COMPLETED'>('IDLE');
  const [errorMessage, setErrorMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [screeningId, setScreeningId] = useState<number | null>(null);
  const [patientDetails, setPatientDetails] = useState({
    patient_name: '',
    patient_age: '',
    patient_gender: '',
    preferred_language: 'en'
  });
  
  // To store the results specifically for the UI
  const [screeningResult, setScreeningResult] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handlePatientSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setCurrentStep('UPLOAD');
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setStatus('IDLE');
      setErrorMessage('');
    }
  };

  // Pass id directly — React setState is async so reading screeningId state
  // inside hitBackendProcess would race and return null.
  const simulateStep = (stepIndex: number, id: number) => {
    if (stepIndex >= PIPELINE_STEPS.length) return;
    setCurrentStep(PIPELINE_STEPS[stepIndex]);

    if (PIPELINE_STEPS[stepIndex] === 'ANALYSIS') {
      hitBackendProcess(id);
      return;
    }

    setTimeout(() => {
      simulateStep(stepIndex + 1, id);
    }, 200);
  };

  const hitBackendProcess = async (id: number) => {
    try {
      setCurrentStep('ANALYSIS');
      const res = await fetch(`${API_BASE_URL}/screening/${id}/process`, { method: 'POST' });
      const data = await res.json();

      if (!data.success) {
        setStatus('ERROR');
        setErrorMessage(data.message || 'MODEL UNAVAILABLE');
        setCurrentStep('ASSESSMENT');
      } else {
        // data.data already has prediction/probabilities/confidence from the process response
        // Fetch full record for any additional fields
        try {
          const fullRes = await fetch(`${API_BASE_URL}/screening/${id}`);
          const fullData = await fullRes.json();
          setScreeningResult(fullData.success ? fullData.data : data.data);
        } catch {
          setScreeningResult(data.data);
        }
        setStatus('COMPLETED');
        setCurrentStep('RESULTS');
      }
    } catch {
      setStatus('ERROR');
      setErrorMessage('Failed to connect to backend.');
      setCurrentStep('ASSESSMENT');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setStatus('PROCESSING');
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('patient_name', patientDetails.patient_name);
    formData.append('patient_age', patientDetails.patient_age);
    formData.append('patient_gender', patientDetails.patient_gender);
    formData.append('preferred_language', patientDetails.preferred_language);

    try {
      const res = await fetch(`${API_BASE_URL}/screening/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      if (data.success) {
        const newId = data.screeningId as number;
        setScreeningId(newId);
        simulateStep(PIPELINE_STEPS.indexOf('QUALITY'), newId);
      } else {
        setStatus('ERROR');
        setErrorMessage('Upload failed.');
      }
    } catch {
      setStatus('ERROR');
      setErrorMessage('Upload failed. Network error.');
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-16 py-24 animate-fade-in">
      <div className="lg:col-span-4 space-y-12">
        <div>
          <h1 className="text-6xl font-light tracking-tighter mb-6">SCREENING</h1>
          <p className="text-xl text-muted-foreground font-light leading-relaxed">
            Upload fundus images for quality assessment, retinal analysis, and DR severity grading.
          </p>
        </div>

        <div className="space-y-4">
          <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">Pipeline Progress</h3>
          <div className="flex flex-col gap-4">
            {PIPELINE_STEPS.map((step, idx) => {
              const isActive = currentStep === step;
              const isPast = PIPELINE_STEPS.indexOf(currentStep) > idx;
              const isError = isActive && status === 'ERROR';

              return (
                <div key={step} className="flex items-center gap-4">
                  <div className={`w-2 h-2 rounded-full ${
                    isError ? 'bg-destructive' : 
                    isActive ? 'bg-primary animate-pulse' : 
                    isPast ? 'bg-muted-foreground' : 'bg-muted'
                  }`} />
                  <span className={`text-sm uppercase tracking-widest ${
                    isError ? 'text-destructive font-bold' :
                    isActive ? 'text-foreground font-bold' : 
                    isPast ? 'text-muted-foreground' : 'text-muted'
                  }`}>
                    {step}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="lg:col-span-8">
        <div className="flex flex-col gap-8">
          
          {/* PATIENT DETAILS */}
          {currentStep === 'PATIENT' && (
            <div className="w-full border border-muted p-8 bg-card/50">
              <h3 className="text-2xl font-light mb-6">Patient Details</h3>
              <form onSubmit={handlePatientSubmit} className="space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-sm uppercase tracking-widest text-muted-foreground">Patient Name</label>
                    <input 
                      type="text" 
                      required
                      className="w-full bg-background border border-muted p-3 text-foreground"
                      value={patientDetails.patient_name}
                      onChange={e => setPatientDetails({...patientDetails, patient_name: e.target.value})}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm uppercase tracking-widest text-muted-foreground">Age</label>
                    <input 
                      type="number" 
                      className="w-full bg-background border border-muted p-3 text-foreground"
                      value={patientDetails.patient_age}
                      onChange={e => setPatientDetails({...patientDetails, patient_age: e.target.value})}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-sm uppercase tracking-widest text-muted-foreground">Gender</label>
                    <select 
                      className="w-full bg-background border border-muted p-3 text-foreground"
                      value={patientDetails.patient_gender}
                      onChange={e => setPatientDetails({...patientDetails, patient_gender: e.target.value})}
                    >
                      <option value="">Select...</option>
                      <option value="M">Male</option>
                      <option value="F">Female</option>
                      <option value="O">Other</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm uppercase tracking-widest text-muted-foreground">Preferred Language</label>
                    <select 
                      className="w-full bg-background border border-muted p-3 text-foreground"
                      value={patientDetails.preferred_language}
                      onChange={e => setPatientDetails({...patientDetails, preferred_language: e.target.value})}
                    >
                      <option value="en">English</option>
                      <option value="hi">हिन्दी</option>
                    </select>
                  </div>
                </div>
                <button type="submit" className="bg-primary text-primary-foreground px-8 py-3 text-sm uppercase tracking-widest hover:opacity-90 mt-4 block w-full text-center">
                  Continue to Upload
                </button>
              </form>
            </div>
          )}

          {/* UPLOAD AREA */}
          {currentStep === 'UPLOAD' && (
            <div className="w-full aspect-video border border-muted border-dashed flex flex-col items-center justify-center p-8 bg-card/50 transition-colors hover:bg-card">
              {previewUrl ? (
                <div className="flex flex-col items-center gap-6 w-full h-full">
                  <img src={previewUrl} alt="Preview" className="h-48 object-cover rounded-sm grayscale" />
                  <p className="text-sm font-bold truncate max-w-sm">{selectedFile?.name}</p>
                  <button 
                    onClick={handleUpload}
                    disabled={status === 'PROCESSING'}
                    className="bg-primary text-primary-foreground px-8 py-3 text-sm uppercase tracking-widest hover:opacity-90 disabled:opacity-50"
                  >
                    {status === 'PROCESSING' ? 'Uploading...' : 'Run Pipeline'}
                  </button>
                  <button onClick={() => setCurrentStep('PATIENT')} className="text-sm underline text-muted-foreground">Back</button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4 text-center cursor-pointer" onClick={() => fileInputRef.current?.click()}>
                  <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                    <span className="text-2xl font-light">+</span>
                  </div>
                  <h3 className="text-2xl font-light">Select Fundus Image</h3>
                  <p className="text-muted-foreground">JPEG, PNG up to 10MB</p>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    accept="image/jpeg, image/png"
                    onChange={handleFileSelect}
                  />
                  <button onClick={() => setCurrentStep('PATIENT')} className="text-sm underline text-muted-foreground mt-4">Back</button>
                </div>
              )}
            </div>
          )}

          {/* PROCESSING / RESULTS VISUALS */}
          {(currentStep !== 'UPLOAD' && currentStep !== 'PATIENT' && currentStep !== 'RESULTS') && (
             <div className="w-full aspect-video border border-muted flex relative overflow-hidden bg-black items-center justify-center">
                {previewUrl && (
                  <img src={previewUrl} alt="Processing" className={`object-cover w-full h-full transition-all duration-1000 ${
                    currentStep === 'QUALITY' ? 'grayscale contrast-125 brightness-90' :
                    currentStep === 'ENHANCEMENT' ? 'grayscale contrast-150 brightness-110' :
                    'grayscale opacity-50'
                  }`} />
                )}
                
                {/* Overlays */}
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 z-10 backdrop-blur-sm p-8 text-center">
                  {status === 'ERROR' ? (
                    <div className="space-y-4 animate-fade-in">
                      <div className="inline-block px-4 py-1 bg-destructive/20 text-destructive border border-destructive text-sm font-bold uppercase tracking-widest mb-4">
                        Pipeline Halted
                      </div>
                      <h2 className="text-4xl tracking-tighter font-light">MODEL ERROR</h2>
                      <p className="text-muted-foreground max-w-md mx-auto">{errorMessage}</p>
                      <div className="flex gap-4 mt-8 justify-center">
                        <button 
                          onClick={() => {
                            setCurrentStep('UPLOAD');
                            setPreviewUrl(null);
                            setSelectedFile(null);
                            setStatus('IDLE');
                            setScreeningId(null);
                          }}
                          className="border border-muted px-6 py-2 text-sm uppercase tracking-widest hover:bg-muted"
                        >
                          Start Over
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4 animate-pulse">
                      <h2 className="text-2xl tracking-widest uppercase font-light text-white">
                        {currentStep === 'QUALITY' ? 'Assessing Quality...' :
                         currentStep === 'ENHANCEMENT' ? 'Applying Filters...' :
                         currentStep === 'ANALYSIS' ? 'Running Model Inference...' : 'Processing...'}
                      </h2>
                    </div>
                  )}
                </div>
             </div>
          )}

          {currentStep === 'RESULTS' && screeningId && (
            <div className="w-full animate-fade-in space-y-8">
              
              {/* Grade + Referral summary */}
              <div className="border border-muted p-8 bg-card/30">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
                  <div>
                    <p className="text-sm font-bold uppercase tracking-widest text-muted-foreground mb-2">AI-Predicted Severity</p>
                    <h2 className="text-5xl font-light mb-2">Grade {screeningResult?.prediction}</h2>
                    <p className="text-xl text-muted-foreground uppercase tracking-widest">
                      {['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR'][screeningResult?.prediction ?? 0]}
                    </p>
                    <p className="text-sm text-muted-foreground mt-3 uppercase tracking-widest">
                      Confidence: {screeningResult?.confidence != null ? (screeningResult.confidence * 100).toFixed(1) : 'N/A'}%
                    </p>
                  </div>
                  <div className="space-y-4">
                    <div className={`px-6 py-4 border-2 text-sm font-bold uppercase tracking-widest ${
                      screeningResult?.referable_dr ? 'border-destructive text-destructive bg-destructive/10' : 'border-emerald-600 text-emerald-600 bg-emerald-600/10'
                    }`}>
                      {screeningResult?.referable_dr ? '⚠ REFERABLE DR — Specialist Review Needed' : '✓ NON-REFERABLE — Routine Care'}
                    </div>
                    {screeningResult?.probabilities && (
                      <div className="space-y-2">
                        {['No DR','Mild DR','Moderate DR','Severe DR','Proliferative DR'].map((label, i) => {
                          const pct = (screeningResult.probabilities[i] * 100);
                          const isBest = screeningResult.prediction === i;
                          return (
                            <div key={i}>
                              <div className="flex justify-between text-xs mb-0.5">
                                <span className={`uppercase tracking-widest ${isBest ? 'text-foreground font-bold' : 'text-muted-foreground'}`}>G{i} {label}</span>
                                <span className={`font-mono ${isBest ? 'text-foreground font-bold' : 'text-muted-foreground'}`}>{pct.toFixed(1)}%</span>
                              </div>
                              <div className="w-full bg-muted h-1.5">
                                <div className={`h-1.5 ${isBest ? 'bg-primary' : 'bg-muted-foreground/40'}`} style={{ width: `${pct}%` }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="w-full h-full bg-background">
                 <ExplainabilityPanel screeningId={screeningId} />
              </div>

              <div className="flex gap-4 border-t border-muted pt-8">
                <Link to={`/reports/${screeningId}/doctor`} className="bg-primary text-primary-foreground px-6 py-3 text-sm uppercase tracking-widest hover:opacity-90 flex-1 text-center">
                  View Doctor Report
                </Link>
                <Link to={`/reports/${screeningId}/patient`} className="bg-background text-foreground border border-muted px-6 py-3 text-sm uppercase tracking-widest hover:bg-muted flex-1 text-center">
                  View Patient Report
                </Link>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Screening;
