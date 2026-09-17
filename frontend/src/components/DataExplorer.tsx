import { useState, useEffect } from 'react';

interface Annotation {
  type: string;
  data: any;
}

interface DatasetImage {
  id: number;
  filename: string;
  relative_path: string;
  dr_grade: number | null;
  annotations: Annotation[];
}

interface Dataset {
  id: string;
  name: string;
  status: string;
}

const API_BASE_URL = 'http://localhost:5000/api';

interface DataExplorerProps {
  onDatasetChange?: (datasetId: string) => void;
}

export const DataExplorer = ({ onDatasetChange }: DataExplorerProps = {}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [activeDataset, setActiveDataset] = useState<string>('');
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE_URL}/datasets`)
      .then(res => res.json())
      .then(data => {
        setDatasets(data);
        if (data.length > 0) setActiveDataset(data[0].id);
      });
  }, []);

  useEffect(() => {
    if (!activeDataset) return;
    if (onDatasetChange) onDatasetChange(activeDataset);
    setLoading(true);
    fetch(`${API_BASE_URL}/datasets/${activeDataset}/images`)
      .then(res => res.json())
      .then(data => {
        setImages(data);
        setLoading(false);
      });
  }, [activeDataset, onDatasetChange]);

  return (
    <div className="flex flex-col gap-8 w-full max-w-7xl">
      <div className="flex gap-4 border-b border-muted pb-4">
        {datasets.map(d => (
          <button
            key={d.id}
            onClick={() => setActiveDataset(d.id)}
            className={`px-4 py-2 uppercase tracking-widest text-sm transition-colors ${
              activeDataset === d.id ? 'text-primary border-b-2 border-primary -mb-[17px]' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {d.name} ({d.status})
          </button>
        ))}
      </div>

      <div className="min-h-[400px]">
        {loading ? (
          <p className="text-muted-foreground uppercase tracking-widest text-sm">Loading samples...</p>
        ) : images.length === 0 ? (
          <p className="text-muted-foreground uppercase tracking-widest text-sm">No images found for this dataset.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {images.slice(0, 12).map(img => (
              <div key={img.id} className="flex flex-col gap-2">
                <div className="bg-muted w-full aspect-square relative overflow-hidden rounded-sm group">
                  <img 
                    src={`${API_BASE_URL}/datasets/${activeDataset}/image/${img.filename}`}
                    alt={img.filename}
                    className="object-cover w-full h-full grayscale group-hover:grayscale-0 transition-all duration-500"
                  />
                  {img.annotations.find(a => a.type === 'VESSEL_MASK') && (
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <p className="text-xs uppercase tracking-widest text-white">Vessel Mask Available</p>
                    </div>
                  )}
                </div>
                <div className="flex justify-between items-start mt-2">
                  <div className="flex flex-col">
                    <span className="text-sm font-bold truncate max-w-[150px]">{img.filename}</span>
                    <span className="text-xs text-muted-foreground uppercase tracking-widest">
                      {img.dr_grade !== null ? `DR Grade: ${img.dr_grade}` : 'No Grade'}
                    </span>
                  </div>
                  <div className="flex flex-col items-end">
                    {img.annotations.map((ann, i) => (
                      <span key={i} className="text-xs text-accent uppercase tracking-widest">
                        {ann.type === 'DME' ? `DME Risk: ${ann.data.risk}` : ann.type}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
