import { useState, useEffect } from 'react';
import { DataExplorer } from '../components/DataExplorer';
import { ValidationDashboard } from '../components/ValidationDashboard';

interface DatasetInfo {
  id: string;
  name: string;
  path: string;
  status: 'CONNECTED' | 'PARTIAL' | 'MISSING' | 'INVALID';
  purpose?: string;
  task?: string;
  source?: string;
  image_count?: number;
  last_scanned?: string;
}

const API_BASE_URL = '/api';

const Validation = () => {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [activeDataset, setActiveDataset] = useState<string>('aptos2019');
  const [scanning, setScanning] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchDatasets = () => {
    setLoading(true);
    fetch(`${API_BASE_URL}/datasets`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setDatasets(data);
          if (data.length > 0 && !activeDataset) {
            setActiveDataset(data[0].id);
          }
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleScanDatasets = async () => {
    setScanning(true);
    try {
      await fetch(`${API_BASE_URL}/datasets/scan`);
      fetchDatasets();
    } catch {
      // scan error handled gracefully
    } finally {
      setScanning(false);
    }
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'CONNECTED':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50';
      case 'PARTIAL':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/50';
      case 'MISSING':
        return 'bg-muted text-muted-foreground border-border';
      default:
        return 'bg-destructive/20 text-destructive border-destructive/50';
    }
  };

  return (
    <div className="py-16 space-y-16 animate-fade-in">
      {/* 1. Header */}
      <section className="flex flex-col md:flex-row justify-between items-start md:items-end gap-8 border-b border-border pb-12">
        <div className="max-w-3xl">
          <h1 className="text-6xl font-light tracking-tighter mb-4">DATASETS & VALIDATION</h1>
          <p className="text-xl text-muted-foreground font-light leading-relaxed">
            Verify model performance across distinct clinical populations. DR-SUGAR evaluates generalization capabilities on globally recognized benchmark datasets without fabricating predictions or confidence scores.
          </p>
        </div>
        <button
          onClick={handleScanDatasets}
          disabled={scanning}
          className="border border-primary text-primary px-6 py-3 uppercase tracking-widest text-sm hover:bg-primary/10 transition-colors disabled:opacity-50 cursor-pointer whitespace-nowrap"
        >
          {scanning ? 'Scanning Filesystem...' : 'Scan / Sync Datasets'}
        </button>
      </section>

      {/* 2. Official Dataset Registry Cards */}
      <section className="w-full space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-light tracking-widest uppercase">Benchmark Datasets</h2>
          <span className="text-xs text-muted-foreground uppercase tracking-widest">
            {datasets.length} Datasets Registered
          </span>
        </div>

        {loading ? (
          <div className="p-12 text-center text-muted-foreground uppercase tracking-widest text-sm animate-pulse">
            Loading dataset registry...
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {datasets.map(ds => {
              const isActive = ds.id === activeDataset;
              return (
                <div
                  key={ds.id}
                  onClick={() => setActiveDataset(ds.id)}
                  className={`border p-6 flex flex-col justify-between transition-all cursor-pointer ${
                    isActive
                      ? 'border-primary bg-card ring-1 ring-primary'
                      : 'border-border bg-card/40 hover:border-foreground/40'
                  }`}
                >
                  <div className="space-y-4">
                    <div className="flex justify-between items-start gap-2">
                      <h3 className="text-xl font-light tracking-wider uppercase text-foreground">{ds.name}</h3>
                      <span className={`px-2.5 py-0.5 text-[10px] uppercase tracking-widest border font-bold ${getStatusBadgeClass(ds.status)}`}>
                        {ds.status}
                      </span>
                    </div>

                    <div className="space-y-2 text-xs">
                      <p className="text-primary uppercase tracking-widest font-semibold">{ds.task || 'Medical Analysis'}</p>
                      <p className="text-muted-foreground leading-relaxed min-h-[48px]">{ds.purpose}</p>
                    </div>
                  </div>

                  <div className="border-t border-border pt-4 mt-6 space-y-2 text-xs text-muted-foreground">
                    <div className="flex justify-between">
                      <span>Indexed Images:</span>
                      <strong className="text-foreground">{ds.image_count ?? 0}</strong>
                    </div>
                    <div className="flex justify-between truncate">
                      <span>Storage Path:</span>
                      <span className="font-mono text-[10px] text-foreground truncate max-w-[120px]" title={ds.path}>
                        data/{ds.id}/
                      </span>
                    </div>
                    {ds.source && (
                      <a
                        href={ds.source}
                        target="_blank"
                        rel="noreferrer"
                        onClick={e => e.stopPropagation()}
                        className="text-[10px] uppercase tracking-widest text-primary hover:underline block pt-1"
                      >
                        Official Source &rarr;
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* 3. Data Explorer */}
      <section className="w-full border-t border-border pt-12">
        <h2 className="text-3xl font-light tracking-tighter mb-8">Data Explorer</h2>
        <DataExplorer selectedDataset={activeDataset} onDatasetChange={setActiveDataset} />
      </section>

      {/* 4. Evaluation & Validation Dashboard */}
      {activeDataset && (
        <section className="w-full border-t border-border pt-12">
          <ValidationDashboard datasetId={activeDataset} />
        </section>
      )}
    </div>
  );
};

export default Validation;
