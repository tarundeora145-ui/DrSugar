import { useState, useEffect } from 'react';

const API_BASE_URL = '/api';

interface ValidationDashboardProps {
  datasetId: string;
}

const GRADE_LABELS = ['Grade 0 (No DR)', 'Grade 1 (Mild)', 'Grade 2 (Moderate)', 'Grade 3 (Severe)', 'Grade 4 (Proliferative)'];

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
      if (result.success && result.data) {
        setData(result.data);
      } else if (result.message) {
        setErrorMsg(result.message);
      }
      fetchLatest();
    } catch (e: any) {
      setErrorMsg('Failed to run validation');
    }
    setRunning(false);
  };

  if (loading) {
    return <div className="text-muted-foreground uppercase tracking-widest text-sm animate-pulse p-8">Loading Validation Benchmark Data...</div>;
  }

  const results = data?.results ? JSON.parse(data.results) : null;
  const isEvaluated = data?.status === 'COMPLETED' && results;
  const stats = data?.stats || data?.meta?.stats;
  const gradeDist = stats?.grade_distribution || {};
  const totalLabeled = stats?.total_labeled || 1;

  const fullCm = results?.full_confusion_matrix;

  return (
    <div className="w-full flex flex-col gap-8 bg-card border border-border p-8 mt-8">
      
      {/* Header & Meta */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-border pb-6">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h2 className="text-2xl font-light uppercase tracking-widest text-foreground">{datasetId} Clinical Benchmark</h2>
            <span className="px-2.5 py-0.5 text-[10px] uppercase tracking-widest font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/50">
              {data?.status || 'NOT EVALUATED'}
            </span>
          </div>
          <p className="text-muted-foreground text-sm space-x-2">
            <span>Model: <strong className="text-foreground">{data?.model_name || data?.model_version || 'DR-SUGAR-V1'}</strong></span>
            <span>&bull;</span>
            <span>Test Split: <strong className="text-foreground">{data?.meta?.evaluation_split || 'split_test.csv'}</strong></span>
            <span>&bull;</span>
            <span>Evaluated Samples: <strong className="text-foreground">{stats?.total_files || 0} images</strong></span>
          </p>
        </div>

        <button 
          onClick={handleRunValidation}
          disabled={running}
          className="border border-primary text-primary px-6 py-3 uppercase tracking-widest text-xs hover:bg-primary/10 disabled:opacity-50 cursor-pointer font-semibold"
        >
          {running ? 'Re-Evaluating Model...' : 'Run Benchmark Evaluation'}
        </button>
      </div>

      {errorMsg && (
        <div className="w-full p-4 bg-destructive/20 text-destructive border border-destructive font-bold uppercase tracking-widest text-sm">
          {errorMsg}
        </div>
      )}

      {/* Actual Dataset Class Distribution (Empirical numbers from SQLite) */}
      {stats && (
        <div className="border border-border p-6 bg-card/50 space-y-4">
          <h3 className="text-xs uppercase tracking-widest text-muted-foreground font-semibold">
            Empirical Class Distribution (SQLite Ground Truth &bull; {stats.total_labeled} Labeled Records)
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[0, 1, 2, 3, 4].map(g => {
              const count = gradeDist[g] || 0;
              const pct = totalLabeled > 0 ? ((count / totalLabeled) * 100).toFixed(1) : '0';
              return (
                <div key={g} className="border border-border p-3 flex flex-col justify-between">
                  <span className="text-[10px] uppercase tracking-widest text-muted-foreground">{GRADE_LABELS[g]}</span>
                  <div className="mt-2">
                    <span className="text-2xl font-light text-foreground">{count}</span>
                    <span className="text-xs text-muted-foreground ml-1">({pct}%)</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Target & Measured Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Sensitivity (Referable DR)', value: results?.sensitivity, note: 'Target > 90%' },
          { label: 'Specificity (Referable DR)', value: results?.specificity, note: 'Target > 85%' },
          { label: 'AUROC (ROC Area)', value: results?.auroc, note: 'Target > 0.90' },
          { label: 'F1 Score (Referable)', value: results?.f1, note: 'Harmonic Mean' }
        ].map((metric, i) => (
           <div key={i} className="border border-border p-5 flex flex-col gap-2">
             <span className="text-xs text-muted-foreground uppercase tracking-widest font-semibold">{metric.label}</span>
             <span className="text-3xl font-light text-foreground">{isEvaluated ? metric.value : '-'}</span>
             <span className="text-[10px] text-accent uppercase tracking-widest">{metric.note}</span>
           </div>
        ))}
      </div>

      {/* Matrices Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-2">
        
        {/* Referable DR Confusion Matrix (2x2) */}
        <div className="border border-border p-6 flex flex-col justify-between">
           <div className="mb-6">
             <h3 className="text-sm uppercase tracking-widest text-foreground font-semibold mb-1">Referable DR Triage Matrix</h3>
             <p className="text-xs text-muted-foreground">Binary triage for Grade &ge; 2 (Moderate, Severe, Proliferative DR requiring ophthalmologist referral).</p>
           </div>
           
           <div className="grid grid-cols-2 gap-4">
             <div className="border border-border p-4 text-center bg-card/50">
               <span className="block text-[10px] uppercase tracking-widest text-muted-foreground mb-1">True Positive (TP)</span>
               <span className="text-3xl font-light text-emerald-400">{isEvaluated ? results?.confusion_matrix?.tp : '-'}</span>
             </div>
             <div className="border border-border p-4 text-center bg-card/50">
               <span className="block text-[10px] uppercase tracking-widest text-muted-foreground mb-1">False Positive (FP)</span>
               <span className="text-3xl font-light text-amber-400">{isEvaluated ? results?.confusion_matrix?.fp : '-'}</span>
             </div>
             <div className="border border-border p-4 text-center bg-card/50">
               <span className="block text-[10px] uppercase tracking-widest text-muted-foreground mb-1">False Negative (FN)</span>
               <span className="text-3xl font-light text-destructive">{isEvaluated ? results?.confusion_matrix?.fn : '-'}</span>
             </div>
             <div className="border border-border p-4 text-center bg-card/50">
               <span className="block text-[10px] uppercase tracking-widest text-muted-foreground mb-1">True Negative (TN)</span>
               <span className="text-3xl font-light text-foreground">{isEvaluated ? results?.confusion_matrix?.tn : '-'}</span>
             </div>
           </div>
        </div>

        {/* Secondary Metrics */}
        <div className="border border-border p-6 flex flex-col justify-between">
           <div>
             <h3 className="text-sm uppercase tracking-widest text-foreground font-semibold mb-1">Comprehensive Model Performance</h3>
             <p className="text-xs text-muted-foreground mb-6">Multi-class evaluation metrics across all 5 severity levels.</p>
           </div>

           <div className="space-y-4">
             <div className="flex justify-between border-b border-border pb-3">
               <span className="text-sm text-muted-foreground">Overall Test Accuracy</span>
               <span className="text-foreground font-semibold">{isEvaluated ? results?.accuracy : 'NOT EVALUATED'}</span>
             </div>
             <div className="flex justify-between border-b border-border pb-3">
               <span className="text-sm text-muted-foreground">Macro Precision</span>
               <span className="text-foreground font-semibold">{isEvaluated ? (results?.macro_precision || results?.precision) : 'NOT EVALUATED'}</span>
             </div>
             <div className="flex justify-between border-b border-border pb-3">
               <span className="text-sm text-muted-foreground">Macro Recall</span>
               <span className="text-foreground font-semibold">{isEvaluated ? (results?.macro_recall || results?.sensitivity) : 'NOT EVALUATED'}</span>
             </div>
             <div className="flex justify-between border-b border-border pb-3">
               <span className="text-sm text-muted-foreground">Macro F1 Score</span>
               <span className="text-foreground font-semibold">{isEvaluated ? (results?.macro_f1 || results?.f1) : 'NOT EVALUATED'}</span>
             </div>
           </div>
        </div>

      </div>

      {/* Full 5x5 Multi-Class Confusion Matrix Table */}
      {fullCm && Array.isArray(fullCm) && (
        <div className="border border-border p-6 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-sm uppercase tracking-widest text-foreground font-semibold">5-Class Severity Confusion Matrix</h3>
            <span className="text-xs text-muted-foreground">Rows = Actual Ground Truth &bull; Columns = Model Predicted</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-center border-collapse">
              <thead>
                <tr className="border-b border-border text-muted-foreground">
                  <th className="p-2 text-left uppercase tracking-widest font-medium">Actual \ Predicted</th>
                  {GRADE_LABELS.map((_, i) => (
                    <th key={i} className="p-2 uppercase tracking-widest font-medium">Gr {i}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {fullCm.map((row: number[], rIdx: number) => (
                  <tr key={rIdx} className="border-b border-border/50 hover:bg-muted/30">
                    <td className="p-3 text-left font-semibold text-foreground uppercase tracking-widest">
                      {GRADE_LABELS[rIdx]}
                    </td>
                    {row.map((val: number, cIdx: number) => {
                      const isDiag = rIdx === cIdx;
                      return (
                        <td
                          key={cIdx}
                          className={`p-3 font-mono text-sm ${
                            isDiag
                              ? 'bg-primary/20 text-primary font-bold border border-primary/40'
                              : val > 0
                              ? 'text-foreground'
                              : 'text-muted-foreground opacity-40'
                          }`}
                        >
                          {val}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
};
