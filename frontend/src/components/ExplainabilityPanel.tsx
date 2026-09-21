import { useEffect, useState } from 'react';

const API_BASE_URL = '/api';
const SERVER_URL = '';

interface EvidenceStatus {
  status: 'PENDING' | 'READY' | 'FAILED';
  url: string | null;
  data: any;
}

interface ScreeningStatus {
  screeningStatus: string;
  gradcam: EvidenceStatus;
  lesion: EvidenceStatus;
  vessel: EvidenceStatus;
}

export const ExplainabilityPanel = ({ screeningId }: { screeningId: number }) => {
  const [data, setData] = useState<ScreeningStatus | null>(null);

  useEffect(() => {
    let interval: number;
    const fetchStatus = () => {
      fetch(`${API_BASE_URL}/screening/${screeningId}/evidence-status`)
        .then(res => res.json())
        .then(resData => {
          if (resData.success) {
            setData(resData);
            if (
              resData.gradcam.status === 'PENDING' ||
              resData.lesion.status === 'PENDING' ||
              resData.vessel.status === 'PENDING'
            ) {
              // keep polling
            } else {
              clearInterval(interval);
            }
          }
        })
        .catch(() => {});
    };

    fetchStatus();
    interval = window.setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [screeningId]);

  if (!data) {
    return <div className="text-muted-foreground uppercase tracking-widest text-sm animate-pulse">Initializing Explainability Engine...</div>;
  }

  const renderPanel = (title: string, ev: EvidenceStatus, emptyMessage: string) => {
    return (
      <div className="space-y-4">
        <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground flex justify-between items-center">
          {title}
          {ev.status === 'PENDING' && <span className="text-primary animate-pulse text-xs">Generating...</span>}
          {ev.status === 'READY' && <span className="text-emerald-500 text-xs">Ready</span>}
        </h3>
        <div className="w-full aspect-square border border-border bg-black relative flex items-center justify-center overflow-hidden">
          {ev.url ? (
            <img src={`${SERVER_URL}${ev.url}`} alt={title} className="w-full h-full object-contain" />
          ) : (
            <div className="space-y-4 z-10 relative text-center p-8">
              {ev.status === 'PENDING' ? (
                <>
                  <div className="inline-block px-4 py-1 bg-primary/20 text-primary border border-primary text-sm font-bold uppercase tracking-widest mb-4 animate-pulse">
                    Processing
                  </div>
                  <h4 className="text-2xl font-light text-white">ANALYZING</h4>
                  <p className="text-muted-foreground text-sm max-w-[250px] mx-auto">Please wait while the AI generates evidence...</p>
                </>
              ) : (
                <>
                  <div className="inline-block px-4 py-1 bg-destructive/20 text-destructive border border-destructive text-sm font-bold uppercase tracking-widest mb-4">
                    Feature Disabled
                  </div>
                  <h4 className="text-2xl font-light text-white">UNAVAILABLE</h4>
                  <p className="text-muted-foreground text-sm max-w-[250px] mx-auto">{emptyMessage}</p>
                </>
              )}
            </div>
          )}
          {!ev.url && (
            <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)', backgroundSize: '16px 16px' }} />
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-8 w-full pb-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Grad-CAM Viewer */}
        <div>
          {renderPanel('Model Activation (Grad-CAM)', data.gradcam, 'No model activations exist.')}
          <div className="text-sm text-muted-foreground mt-2">
            Highlights regions most heavily weighted by the DR classifier.
          </div>
        </div>

        {/* Lesion Segmentation */}
        <div>
          {renderPanel('Lesion Evidence (IDRiD V3)', data.lesion, 'LESION SEGMENTATION UNAVAILABLE')}
          {data.lesion.data && (
            <div className="space-y-1 mt-2">
              <p className="text-[10px] text-muted-foreground uppercase tracking-widest">
                % of image area (IDRiD V3, Fold 1)
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs uppercase tracking-widest">
                <div className="border border-border p-2 flex justify-between"><span className="text-red-500 font-bold">MA:</span> <span>{data.lesion.data.MA}%</span></div>
                <div className="border border-border p-2 flex justify-between"><span className="text-orange-500 font-bold">HE:</span> <span>{data.lesion.data.HE}%</span></div>
                <div className="border border-border p-2 flex justify-between"><span className="text-yellow-500 font-bold">EX:</span> <span>{data.lesion.data.EX}%</span></div>
                <div className="border border-border p-2 flex justify-between"><span className="text-cyan-500 font-bold">SE:</span> <span>{data.lesion.data.SE}%</span></div>
              </div>
            </div>
          )}
        </div>

        {/* Vessel Segmentation */}
        <div>
          {renderPanel('Vascular Structure (DRIVE V1)', data.vessel, 'VESSEL SEGMENTATION UNAVAILABLE')}
          {data.vessel.data && (
            <div className="mt-2 border border-border p-3 flex justify-between text-xs uppercase tracking-widest">
              <span className="font-bold">Total Vessel Coverage:</span>
              <span>{data.vessel.data.coverage}%</span>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};
