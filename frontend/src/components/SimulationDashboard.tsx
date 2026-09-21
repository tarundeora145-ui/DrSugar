import { useState } from 'react';

export const SimulationDashboard = () => {
  const [params, setParams] = useState({
    patientsPerYear: 5000,
    imagesPerPatient: 2,
    imageSizeMB: 4.5,
    bandwidthMbps: 2.0,
    processingTimeS: 0.8,
    reviewCapacity: 15,
    referralRate: 0.15
  });

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setParams({ ...params, [e.target.name]: parseFloat(e.target.value) || 0 });
  };

  const handleRun = async () => {
    setRunning(true);
    setResult(null);
    setErrorMsg(null);
    try {
      const res = await fetch('/api/simulation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const data = await res.json();
      if (data.success) {
        setResult(data.simulation);
      } else {
        setErrorMsg(data.error || 'Simulation execution failed.');
      }
    } catch (e: any) {
      setErrorMsg(e.message || 'Failed to connect to simulation engine.');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="w-full flex flex-col gap-8">
      
      {/* Simulation Network Diagram */}
      <div className="w-full border border-border p-8 bg-card">
        <h3 className="text-sm uppercase tracking-widest text-muted-foreground mb-8">Rural Deployment Network Architecture</h3>
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-center">
          <div className="border border-border p-4 w-full md:w-auto bg-card/50">
            <span className="block text-xs uppercase tracking-widest text-muted-foreground">Fundus Camera</span>
            <span className="text-xl font-light">PHC</span>
          </div>
          <span className="text-muted-foreground">&rarr;</span>
          <div className="border border-border p-4 w-full md:w-auto bg-card/50">
            <span className="block text-xs uppercase tracking-widest text-muted-foreground">Quality Gate</span>
            <span className="text-xl font-light">Edge Node</span>
          </div>
          <span className="text-muted-foreground">&rarr;</span>
          <div className="border border-border p-4 w-full md:w-auto bg-card/50">
            <span className="block text-xs uppercase tracking-widest text-muted-foreground">Grading & Explainability</span>
            <span className="text-xl font-light text-primary">DR-SUGAR AI</span>
          </div>
          <span className="text-muted-foreground">&rarr;</span>
          <div className="border border-border p-4 w-full md:w-auto bg-card/50">
            <span className="block text-xs uppercase tracking-widest text-muted-foreground">Teleophthalmology</span>
            <span className="text-xl font-light text-accent">Specialist Review</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Parameters Input */}
        <div className="border border-border p-8 bg-card flex flex-col gap-6">
          <h3 className="text-sm uppercase tracking-widest text-muted-foreground">Simulation Parameters</h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Patients / Year</label>
              <input type="number" name="patientsPerYear" value={params.patientsPerYear} onChange={handleChange} className="bg-transparent border border-border p-2 font-light text-foreground" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Images / Patient</label>
              <input type="number" name="imagesPerPatient" value={params.imagesPerPatient} onChange={handleChange} className="bg-transparent border border-border p-2 font-light text-foreground" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Image Size (MB)</label>
              <input type="number" name="imageSizeMB" value={params.imageSizeMB} onChange={handleChange} className="bg-transparent border border-border p-2 font-light text-foreground" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Bandwidth (Mbps)</label>
              <input type="number" name="bandwidthMbps" value={params.bandwidthMbps} onChange={handleChange} className="bg-transparent border border-border p-2 font-light text-foreground" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">AI Processing (sec)</label>
              <input type="number" name="processingTimeS" value={params.processingTimeS} onChange={handleChange} className="bg-transparent border border-border p-2 font-light text-foreground" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Review Capacity</label>
              <input type="number" name="reviewCapacity" value={params.reviewCapacity} onChange={handleChange} className="bg-transparent border border-border p-2 font-light text-foreground" />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs uppercase tracking-widest text-muted-foreground">Referral Rate</label>
              <input type="number" name="referralRate" value={params.referralRate} step="0.01" onChange={handleChange} className="bg-transparent border border-border p-2 font-light text-foreground" />
            </div>
          </div>

          <button 
            onClick={handleRun}
            disabled={running}
            className="mt-4 border border-primary text-primary px-8 py-3 uppercase tracking-widest text-sm hover:bg-primary/10 transition-colors disabled:opacity-50 cursor-pointer"
          >
            {running ? 'Running Simulation...' : 'Execute Queue Simulation'}
          </button>
        </div>

        {/* Results Panel */}
        <div className="border border-border p-8 bg-card flex flex-col gap-6 relative overflow-hidden">
          <h3 className="text-sm uppercase tracking-widest text-muted-foreground">Simulation Results</h3>
          
          {errorMsg ? (
            <div className="space-y-4 z-10 relative mt-8">
              <div className="inline-block px-4 py-1 bg-destructive/20 text-destructive border border-destructive text-sm font-bold uppercase tracking-widest mb-4">
                ERROR
              </div>
              <h4 className="text-2xl font-light text-foreground leading-tight">{errorMsg}</h4>
            </div>
          ) : result ? (
            <div className="space-y-6 z-10 relative mt-4">
              <div className="inline-block px-4 py-1 bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 text-sm font-bold uppercase tracking-widest mb-2">
                SIMULATION COMPLETE
              </div>

              <div className="grid grid-cols-2 gap-6 pt-2">
                <div>
                  <span className="block text-xs uppercase tracking-widest text-muted-foreground">Network Bandwidth</span>
                  <span className="text-3xl font-light">{result.metrics.bandwidthMbps} Mbps</span>
                </div>
                <div>
                  <span className="block text-xs uppercase tracking-widest text-muted-foreground">Total Data / Year</span>
                  <span className="text-3xl font-light">{result.metrics.totalDataGB} GB</span>
                </div>
                <div>
                  <span className="block text-xs uppercase tracking-widest text-muted-foreground">Upload Time / Image</span>
                  <span className="text-3xl font-light text-accent">{result.metrics.uploadTimeSec}s</span>
                </div>
                <div>
                  <span className="block text-xs uppercase tracking-widest text-muted-foreground">Total Turnaround</span>
                  <span className="text-3xl font-light text-primary">{result.metrics.turnaroundSec}s</span>
                </div>
              </div>

              <div className="border-t border-border pt-6 mt-6 space-y-3">
                <p className="text-sm text-muted-foreground">
                  <strong className="text-foreground">Referral Backlog:</strong> {result.summary.referredCasesPerYear} cases/year automatically triaged for specialist review.
                </p>
                <p className="text-sm text-muted-foreground">
                  <strong className="text-foreground">Review Workload:</strong> ~{result.summary.dailySpecialistLoad} cases/day required for ophthalmologists.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center text-muted-foreground p-12 space-y-4">
              <p className="text-sm uppercase tracking-widest">No Active Run</p>
              <p className="text-xs max-w-xs">Adjust network & patient parameters on the left and click execute to model queue performance.</p>
            </div>
          )}
        </div>

      </div>

    </div>
  );
};
