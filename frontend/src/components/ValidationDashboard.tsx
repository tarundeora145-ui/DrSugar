import { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:5000/api';

interface ValidationDashboardProps {
  datasetId: string;
}

export const ValidationDashboard = ({ datasetId }: ValidationDashboardProps) => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any>(null);
  const [running, setRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const fetchLatest = () => {
    setLoading(true);
    fetch(`${API_BASE_URL}/validation/${datasetId}/latest`)
      .then(res => res.json())
      .then(resData => {
        if (resData.success) {
          setData(resData.data);
        }
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchLatest();
  }, [datasetId]);

  const handleRunValidation = async () => {
    setRunning(true);
    setErrorMsg('');
    try {
      const res = await fetch(`${API_BASE_URL}/validation/${datasetId}/run`, {
        method: 'POST'
      });
      const result = await res.json();
      if (!result.success) {
        setErrorMsg(result.message);
      }
      fetchLatest();
    } catch (e: any) {
      setErrorMsg('Failed to run validation');
    }
    setRunning(false);
  };

  if (loading) {
    return <div className="text-muted-foreground uppercase tracking-widest text-sm animate-pulse p-8">Loading Validation Data...</div>;
  }

  const results = data?.results ? JSON.parse(data.results) : null;
  const isEvaluated = data?.status === 'COMPLETED' && results;

  return (
    <div className="w-full flex flex-col gap-8 bg-black border border-muted p-8 mt-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-muted pb-4">
        <div>
          <h2 className="text-2xl font-light uppercase tracking-widest">{datasetId} Validation</h2>
          <p className="text-muted-foreground text-sm">
            Model: <span className="text-white font-medium">{data?.model_version || 'NONE'}</span> &bull; Status: <span className="text-white font-medium">{data?.status || 'NOT EVALUATED'}</span>
          </p>
        </div>
        <button 
          onClick={handleRunValidation}
          disabled={running}
          className="border border-primary text-primary px-6 py-2 uppercase tracking-widest text-sm hover:bg-primary/10 disabled:opacity-50"
        >
          {running ? 'Running...' : 'Run Evaluation'}
        </button>
      </div>

      {errorMsg && (
        <div className="w-full p-4 bg-destructive/20 text-destructive border border-destructive font-bold uppercase tracking-widest text-sm">
          {errorMsg}
        </div>
      )}

      {/* Target Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Sensitivity', value: results?.sensitivity, target: '> 90%' },
          { label: 'Specificity', value: results?.specificity, target: '> 85%' },
          { label: 'AUROC', value: results?.auroc, target: 'Maximize' },
          { label: 'F1 Score', value: results?.f1, target: 'Maximize' }
        ].map((metric, i) => (
           <div key={i} className="border border-muted p-4 flex flex-col gap-2">
             <span className="text-xs text-muted-foreground uppercase tracking-widest">{metric.label}</span>
             <span className="text-3xl font-light">{isEvaluated ? metric.value : '-'}</span>
             <span className="text-xs text-amber-500 uppercase tracking-widest">Target: {metric.target}</span>
           </div>
        ))}
      </div>

      {/* Confusion Matrix (Referable DR) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-4">
        
        <div className="border border-muted p-6">
           <h3 className="text-sm uppercase tracking-widest text-muted-foreground mb-6">Confusion Matrix (Referable DR Level ≥ 2)</h3>
           <div className="grid grid-cols-2 gap-4">
             <div className="border border-muted p-4 text-center bg-card/20">
               <span className="block text-xs uppercase tracking-widest text-muted-foreground mb-2">True Positive</span>
               <span className="text-2xl font-light">{isEvaluated ? results?.confusion_matrix?.tp : '-'}</span>
             </div>
             <div className="border border-muted p-4 text-center bg-card/20">
               <span className="block text-xs uppercase tracking-widest text-muted-foreground mb-2">False Positive</span>
               <span className="text-2xl font-light">{isEvaluated ? results?.confusion_matrix?.fp : '-'}</span>
             </div>
             <div className="border border-muted p-4 text-center bg-card/20">
               <span className="block text-xs uppercase tracking-widest text-muted-foreground mb-2">False Negative</span>
               <span className="text-2xl font-light">{isEvaluated ? results?.confusion_matrix?.fn : '-'}</span>
             </div>
             <div className="border border-muted p-4 text-center bg-card/20">
               <span className="block text-xs uppercase tracking-widest text-muted-foreground mb-2">True Negative</span>
               <span className="text-2xl font-light">{isEvaluated ? results?.confusion_matrix?.tn : '-'}</span>
             </div>
           </div>
        </div>

        <div className="border border-muted p-6">
           <h3 className="text-sm uppercase tracking-widest text-muted-foreground mb-6">Secondary Metrics</h3>
           <div className="space-y-4">
             <div className="flex justify-between border-b border-muted pb-2">
               <span className="text-sm text-muted-foreground">Accuracy</span>
               <span>{isEvaluated ? results?.accuracy : 'NOT EVALUATED'}</span>
             </div>
             <div className="flex justify-between border-b border-muted pb-2">
               <span className="text-sm text-muted-foreground">Precision</span>
               <span>{isEvaluated ? results?.precision : 'NOT EVALUATED'}</span>
             </div>
             <div className="flex justify-between border-b border-muted pb-2">
               <span className="text-sm text-muted-foreground">Recall</span>
               <span>{isEvaluated ? results?.recall : 'NOT EVALUATED'}</span>
             </div>
           </div>
        </div>

      </div>

    </div>
  );
};
