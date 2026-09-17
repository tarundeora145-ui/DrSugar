import { useState } from 'react';

const API_BASE_URL = 'http://localhost:5000/api';

export const SimulationDashboard = () => {
  const [running, setRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  
  // Default to a 100,000+ scenario
  const [params, setParams] = useState({
    patientsPerYear: 100000,
    imagesPerPatient: 2,
    imageSizeMB: 5,
    bandwidthMbps: 2,
    processingTimeS: 4.5,
    reviewCapacity: 15000,
    referralRate: 0.15
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setParams(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  const handleRun = async () => {
    setRunning(true);
    setErrorMsg('');
    try {
      const res = await fetch(`${API_BASE_URL}/simulation/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const data = await res.json();
      if (!data.success) {
        setErrorMsg(data.message);
      }
    } catch (err) {
      setErrorMsg('Failed to connect to simulation server.');
    }
    setRunning(false);
  };

  return (
    <div className="w-full flex flex-col gap-8">
      
      {/* Simulation Network Diagram */}
      <div className="w-full border border-muted p-8 bg-black">
        <h3 className="text-sm uppercase tracking-widest text-muted-foreground mb-8">Rural Deployment Network Architecture</h3>
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-center">
          <div className="border border-muted p-4 w-full md:w-auto bg-card/20">
            <span className="block text-xs uppercase tracking-widest text-muted-foreground">Fundus Camera</span>
            <span className="text-xl font-light">PHC</span>
          </div>
          <span className="text-muted-foreground">&rarr;</span>
          <div className="border border-muted p-4 w-full md:w-auto bg-card/20">
            <span className="block text-xs uppercase tracking-widest text-muted-foreground">Quality Gate</span>
            <span className="text-xl font-light">Edge AI</span>
          </div>
          <span className="text-muted-foreground">&rarr;</span>
          <div className="border border-accent p-4 w-full md:w-auto bg-accent/10">
            <span className="block text-xs uppercase tracking-widest text-muted-foreground">Network</span>
            <span className="text-xl font-light">{params.bandwidthMbps} Mbps</span>
          </div>
          <span className="text-muted-foreground">&rarr;</span>
          <div className="border border-muted p-4 w-full md:w-auto bg-card/20">
            <span className="block text-xs uppercase tracking-widest text-muted-foreground">Central System</span>
            <span className="text-xl font-light">Cloud</span>
          </div>
          <span className="text-muted-foreground">&rarr;</span>
          <div className="border border-muted p-4 w-full md:w-auto bg-card/20">
            <span className="block text-xs uppercase tracking-widest text-muted-foreground">Review</span>
            <span className="text-xl font-light">Ophthalmologist</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Parameters Input */}
        <div className="border border-muted p-8 bg-black flex flex-col gap-6">
          <h3 className="text-sm uppercase tracking-widest text-muted-foreground">Simulation Parameters</h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Patients / Year</label>
              <input type="number" name="patientsPerYear" value={params.patientsPerYear} onChange={handleChange} className="bg-transparent border border-muted p-2 font-light" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Images / Patient</label>
              <input type="number" name="imagesPerPatient" value={params.imagesPerPatient} onChange={handleChange} className="bg-transparent border border-muted p-2 font-light" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Image Size (MB)</label>
              <input type="number" name="imageSizeMB" value={params.imageSizeMB} onChange={handleChange} className="bg-transparent border border-muted p-2 font-light" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Bandwidth (Mbps)</label>
              <input type="number" name="bandwidthMbps" value={params.bandwidthMbps} onChange={handleChange} className="bg-transparent border border-muted p-2 font-light" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">AI Processing (sec)</label>
              <input type="number" name="processingTimeS" value={params.processingTimeS} onChange={handleChange} className="bg-transparent border border-muted p-2 font-light" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Review Capacity</label>
              <input type="number" name="reviewCapacity" value={params.reviewCapacity} onChange={handleChange} className="bg-transparent border border-muted p-2 font-light" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Referral Rate</label>
              <input type="number" name="referralRate" value={params.referralRate} step="0.01" onChange={handleChange} className="bg-transparent border border-muted p-2 font-light" />
            </div>
          </div>

          <button 
            onClick={handleRun}
            disabled={running}
            className="mt-4 border border-primary text-primary px-8 py-3 uppercase tracking-widest text-sm hover:bg-primary/10 transition-colors disabled:opacity-50"
          >
            {running ? 'Running Simulation...' : 'Execute Queue Simulation'}
          </button>
        </div>

        {/* Results Panel */}
        <div className="border border-muted p-8 bg-black flex flex-col gap-6 relative overflow-hidden">
          <h3 className="text-sm uppercase tracking-widest text-muted-foreground">Simulation Results</h3>
          
          {errorMsg ? (
            <div className="space-y-4 z-10 relative mt-8">
              <div className="inline-block px-4 py-1 bg-destructive/20 text-destructive border border-destructive text-sm font-bold uppercase tracking-widest mb-4">
                MATLAB REQUIRED
              </div>
              <h4 className="text-2xl font-light text-white leading-tight">{errorMsg}</h4>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center opacity-50">
              <p className="text-muted-foreground uppercase tracking-widest">Awaiting execution...</p>
            </div>
          )}

          {errorMsg && (
            <div className="absolute inset-0 opacity-5" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)', backgroundSize: '24px 24px' }} />
          )}
        </div>

      </div>
    </div>
  );
};
