
import { useState } from 'react';
import { DataExplorer } from '../components/DataExplorer';
import { ValidationDashboard } from '../components/ValidationDashboard';

const Validation = () => {
  const [activeDataset, setActiveDataset] = useState<string>('');
  return (
    <div className="py-24 space-y-16 animate-fade-in">
      <section className="max-w-2xl">
        <h1 className="text-6xl font-light tracking-tighter mb-6">DATASETS + VALIDATION</h1>
        <p className="text-xl text-muted-foreground font-light leading-relaxed">
          Verify model performance across distinct populations. DR-SUGAR evaluates generalization capabilities on globally recognized datasets without fabricating predictions.
        </p>
      </section>

      <section className="w-full">
        <h2 className="text-3xl font-light tracking-tighter mb-8">Data Explorer</h2>
        <DataExplorer onDatasetChange={setActiveDataset} />
      </section>

      {activeDataset && (
        <section className="w-full mt-12">
          <ValidationDashboard datasetId={activeDataset} />
        </section>
      )}
    </div>
  );
};

export default Validation;
