import { useState, useRef } from 'react';
import { ExplainabilityPanel } from '../components/ExplainabilityPanel';

type Step = 'UPLOAD' | 'QUALITY' | 'ENHANCEMENT' | 'ANALYSIS' | 'ASSESSMENT' | 'EXPLAINABILITY' | 'REPORT';

const PIPELINE_STEPS: Step[] = [
  'UPLOAD', 'QUALITY', 'ENHANCEMENT', 'ANALYSIS', 'ASSESSMENT', 'EXPLAINABILITY', 'REPORT'
];

const API_BASE_URL = 'http://localhost:5000/api';

const Screening = () => {
  const [currentStep, setCurrentStep] = useState<Step>('UPLOAD');
  const [status, setStatus] = useState<'IDLE' | 'PROCESSING' | 'ERROR'>('IDLE');
  const [errorMessage, setErrorMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [screeningId, setScreeningId] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setStatus('IDLE');
      setErrorMessage('');
    }
  };

  const simulateStep = (stepIndex: number) => {
    if (stepIndex >= PIPELINE_STEPS.length) return;
    
    setCurrentStep(PIPELINE_STEPS[stepIndex]);

    if (PIPELINE_STEPS[stepIndex] === 'ANALYSIS') {
      // The moment we hit Analysis, we ping the backend which needs the AI model
      hitBackendProcess();
      return;
    }

    setTimeout(() => {
      simulateStep(stepIndex + 1);
    }, 1500); // simulate 1.5s per non-AI step
  };

  const hitBackendProcess = async () => {
    if (!screeningId) return;
    try {
      const res = await fetch(`${API_BASE_URL}/screening/${screeningId}/process`, { method: 'POST' });
      const data = await res.json();
      
      if (!data.success && data.status === 'MODEL UNAVAILABLE') {
        setStatus('ERROR');
        setErrorMessage(data.message || 'MODEL UNAVAILABLE');
        setCurrentStep('ASSESSMENT');
      }
    } catch (err) {
      setStatus('ERROR');
      setErrorMessage('Failed to connect to backend.');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setStatus('PROCESSING');
    
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const res = await fetch(`${API_BASE_URL}/screening/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      if (data.success) {
        setScreeningId(data.screeningId);
        simulateStep(1); // Start quality assessment
      } else {
        setStatus('ERROR');
        setErrorMessage('Upload failed.');
      }
    } catch (err) {
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
          {/* Upload Area */}
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
                </div>
              )}
            </div>
          )}

          {/* Processing Visuals */}
          {currentStep !== 'UPLOAD' && (
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
                      <h2 className="text-4xl tracking-tighter font-light">MODEL UNAVAILABLE</h2>
                      <p className="text-muted-foreground max-w-md mx-auto">{errorMessage}</p>
                      <div className="flex gap-4 mt-8">
                        <button 
                          onClick={() => {
                            setCurrentStep('EXPLAINABILITY');
                            setStatus('IDLE');
                          }}
                          className="border border-muted px-6 py-2 text-sm uppercase tracking-widest hover:bg-muted"
                        >
                          View XAI Architecture
                        </button>
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
                  ) : currentStep === 'EXPLAINABILITY' && screeningId ? (
                    <div className="w-full h-full bg-background overflow-y-auto">
                       <ExplainabilityPanel screeningId={screeningId} />
                    </div>
                  ) : (
                    <div className="space-y-4 animate-pulse">
                      <h2 className="text-2xl tracking-widest uppercase font-light text-white">
                        {currentStep === 'QUALITY' ? 'Assessing Quality...' :
                         currentStep === 'ENHANCEMENT' ? 'Applying Filters...' :
                         currentStep === 'ANALYSIS' ? 'Connecting to Model...' : 'Processing...'}
                      </h2>
                    </div>
                  )}
                </div>
             </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Screening;
