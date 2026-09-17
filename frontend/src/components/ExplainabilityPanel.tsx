import { useEffect, useState } from 'react';

const API_BASE_URL = 'http://localhost:5000/api';

interface ExplainabilityData {
  gradCamStatus: string;
  lesionEvidence: any;
  vesselMask: any;
  landmarks: any;
  confidence: any;
  message: string;
}

export const ExplainabilityPanel = ({ screeningId }: { screeningId: number }) => {
  const [data, setData] = useState<ExplainabilityData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE_URL}/screening/${screeningId}/explainability`)
      .then(res => res.json())
      .then(resData => {
        setData(resData);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [screeningId]);

  if (loading) {
    return <div className="text-muted-foreground uppercase tracking-widest text-sm animate-pulse">Loading Explainability Data...</div>;
  }

  if (!data) {
    return <div className="text-destructive font-bold uppercase tracking-widest text-sm">Error Loading Explainability</div>;
  }

  return (
    <div className="flex flex-col gap-8 w-full">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Grad-CAM Viewer */}
        <div className="space-y-4">
          <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">Model Activation (Grad-CAM)</h3>
          <div className="w-full aspect-square border border-muted bg-black relative flex items-center justify-center p-8 text-center overflow-hidden">
            {data.gradCamStatus === 'GRAD-CAM UNAVAILABLE' ? (
              <div className="space-y-4 z-10 relative">
                <div className="inline-block px-4 py-1 bg-destructive/20 text-destructive border border-destructive text-sm font-bold uppercase tracking-widest mb-4">
                  Feature Disabled
                </div>
                <h4 className="text-2xl font-light text-white">GRAD-CAM UNAVAILABLE</h4>
                <p className="text-muted-foreground text-sm max-w-[250px] mx-auto">{data.message}</p>
              </div>
            ) : (
              <div className="text-white">Grad-CAM Overlay</div>
            )}
            
            {/* Background pattern for unavailable state */}
            {data.gradCamStatus === 'GRAD-CAM UNAVAILABLE' && (
              <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)', backgroundSize: '16px 16px' }} />
            )}
          </div>
        </div>

        {/* Evidence & Confidence Metrics */}
        <div className="space-y-8 flex flex-col justify-between">
          
          <div className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">Lesion Evidence</h3>
            <div className="border border-muted p-4 bg-card/30">
              {data.lesionEvidence ? (
                <div>Lesion Data Available</div>
              ) : (
                <div className="text-destructive uppercase tracking-widest text-sm font-bold">Evidence Unavailable</div>
              )}
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">Anatomical Landmarks</h3>
            <div className="border border-muted p-4 bg-card/30">
              {data.landmarks ? (
                <div>Landmark Data Available</div>
              ) : (
                <div className="text-destructive uppercase tracking-widest text-sm font-bold">Landmarks Unavailable</div>
              )}
            </div>
          </div>

          <div className="space-y-4">
             <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">Model Confidence</h3>
             <div className="border border-muted p-4 bg-card/30 border-l-4 border-l-destructive">
               <p className="text-xs text-muted-foreground mb-2">CLINICAL CALIBRATION</p>
               {data.confidence ? (
                 <div>{data.confidence}</div>
               ) : (
                 <div className="text-destructive uppercase tracking-widest text-sm font-bold flex items-center gap-2">
                   ⚠️ UNCALIBRATED / UNAVAILABLE
                 </div>
               )}
             </div>
          </div>

        </div>
      </div>
    </div>
  );
};
