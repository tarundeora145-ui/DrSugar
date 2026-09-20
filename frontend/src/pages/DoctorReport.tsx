import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ExplainabilityPanel } from '../components/ExplainabilityPanel';

const API_BASE_URL = 'http://localhost:5000/api';
const SERVER_URL = 'http://localhost:5000';

const DR_LABELS = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR'];

const DoctorReport = () => {
  const { id } = useParams();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API_BASE_URL}/screening/${id}`)
      .then(res => res.json())
      .then(resData => {
        if (resData.success) {
          setData(resData.data);
        } else {
          setError('Screening record not found.');
        }
        setLoading(false);
      })
      .catch(() => {
        setError('Failed to connect to server.');
        setLoading(false);
      });
  }, [id]);

  if (loading) return <div className="p-24 animate-pulse text-gray-600">Loading Report...</div>;
  if (error || !data) return <div className="p-24 text-red-600 font-bold">{error || 'Report not found.'}</div>;

  const probs: number[] = Array.isArray(data.probabilities) ? data.probabilities : [];

  return (
    <div className="max-w-5xl mx-auto py-12 px-8 bg-white text-black min-h-screen print:py-4 print:px-4">
      {/* Print styles */}
      <style>{`
        @media print {
          .print\\:hidden { display: none !important; }
          body { background: white !important; color: black !important; }
          img { max-width: 100%; page-break-inside: avoid; }
        }
      `}</style>

      {/* Header */}
      <div className="flex justify-between items-start border-b-2 border-black pb-8 mb-8">
        <div>
          <h1 className="text-4xl font-black tracking-tighter uppercase">DR-SUGAR</h1>
          <p className="text-sm tracking-widest font-bold uppercase mt-1">Clinical Screening Report</p>
        </div>
        <div className="text-right">
          <button
            onClick={() => window.print()}
            className="bg-black text-white px-6 py-2 text-sm uppercase tracking-widest hover:bg-black/80 print:hidden"
          >
            Print / Save PDF
          </button>
          <div className="mt-4 text-xs font-mono uppercase">
            <p>Screening ID: {data.display_id || data.id}</p>
            <p>Date: {new Date(data.created_at).toLocaleDateString('en-GB', { timeZone: 'Asia/Kolkata' })}</p>
            <p>Time: {new Date(data.created_at).toLocaleTimeString('en-GB', { timeZone: 'Asia/Kolkata' })}</p>
          </div>
        </div>
      </div>

      {/* Patient Info + Original Image */}
      <div className="grid grid-cols-3 gap-8 mb-12">
        <div className="col-span-2 border border-black p-6">
          <h3 className="text-xs uppercase tracking-widest font-bold mb-4 border-b border-black/20 pb-2">Patient Details</h3>
          <div className="grid grid-cols-2 gap-y-3 text-sm">
            <span className="font-bold">Name:</span>
            <span>{data.patient_name || 'N/A'}</span>
            <span className="font-bold">Age:</span>
            <span>{data.patient_age || 'N/A'}</span>
            <span className="font-bold">Gender:</span>
            <span>{data.patient_gender === 'M' ? 'Male' : data.patient_gender === 'F' ? 'Female' : data.patient_gender || 'N/A'}</span>
            <span className="font-bold">Patient ID:</span>
            <span>{data.patient_id || 'N/A'}</span>
            <span className="font-bold">Model Version:</span>
            <span>{data.model_version || 'DR-SUGAR-V1'}</span>
            <span className="font-bold">Pipeline Status:</span>
            <span>{data.status}</span>
          </div>
        </div>
        <div className="border border-black overflow-hidden flex flex-col">
          <p className="text-xs font-bold uppercase tracking-widest px-3 py-2 border-b border-black/20">Fundus Image</p>
          {data.image_path ? (
            <img
              src={`${SERVER_URL}/uploads/${data.image_path}`}
              alt="Original fundus"
              className="w-full h-full object-cover flex-1"
              style={{ maxHeight: 180 }}
            />
          ) : (
            <div className="flex-1 bg-gray-100 flex items-center justify-center text-xs text-gray-400 uppercase tracking-widest">
              Image Unavailable
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      <div className="mb-12">
        <h2 className="text-2xl font-black uppercase mb-6">AI Assessment Results</h2>
        <div className="grid grid-cols-2 gap-8">
          {/* Grade + Referral */}
          <div className="space-y-4">
            <div className="border border-black p-6 bg-black text-white">
              <h3 className="text-xs uppercase tracking-widest mb-3 opacity-70">Predicted DR Severity</h3>
              <p className="text-5xl font-light mb-2">Grade {data.result_grade}</p>
              <p className="text-lg font-bold tracking-widest uppercase">
                {DR_LABELS[data.result_grade ?? 0]}
              </p>
            </div>
            <div className={`border-2 p-6 ${data.referable_dr ? 'border-red-600 text-red-600' : 'border-green-700 text-green-700'}`}>
              <h3 className="text-xs uppercase tracking-widest mb-2 font-bold">Referral Decision</h3>
              <p className="text-3xl font-black mb-2 uppercase">
                {data.referable_dr ? 'REFERABLE' : 'NON-REFERABLE'}
              </p>
              <p className="text-sm font-bold uppercase tracking-widest">
                Model Confidence: {data.confidence != null ? (data.confidence * 100).toFixed(1) : 'N/A'}%
              </p>
              <p className="text-xs mt-2 opacity-70">Grade ≥ 2 → Referable</p>
            </div>
          </div>

          {/* Probabilities */}
          <div className="border border-black p-6">
            <h3 className="text-xs uppercase tracking-widest font-bold mb-4 border-b border-black/20 pb-2">Grade Probability Distribution</h3>
            <div className="space-y-3">
              {DR_LABELS.map((label, i) => {
                const pct = probs[i] != null ? (probs[i] * 100) : 0;
                const isBest = data.result_grade === i;
                return (
                  <div key={i}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className={`font-bold uppercase tracking-widest ${isBest ? 'text-black' : 'text-gray-500'}`}>
                        Grade {i} — {label}
                      </span>
                      <span className={`font-mono font-bold ${isBest ? 'text-black' : 'text-gray-500'}`}>
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-100 h-2">
                      <div
                        className={`h-2 ${isBest ? 'bg-black' : 'bg-gray-300'}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Explainability */}
      <div className="mb-12">
        <h2 className="text-2xl font-black uppercase mb-6 border-b-2 border-black pb-2">
          Explainability & Evidence
        </h2>
        <ExplainabilityPanel screeningId={Number(id)} />
      </div>

      {/* Disclaimer + Signatures */}
      <div className="border border-black/30 p-6 mb-12 bg-gray-50 text-xs text-gray-500 leading-relaxed">
        <strong className="text-black uppercase tracking-widest block mb-2">Clinical Disclaimer</strong>
        This report was generated by an AI-assisted screening prototype (DR-SUGAR). It does not constitute a clinical diagnosis.
        Predictions are research outputs only and require review by a qualified ophthalmologist. The system has not been
        clinically validated or regulatory approved.
      </div>

      <div className="grid grid-cols-2 gap-16">
        <div className="border-t border-black pt-4">
          <p className="text-xs uppercase tracking-widest font-bold">Reviewing Ophthalmologist</p>
          <p className="text-xs mt-2 text-black/60">Signature &amp; Date</p>
        </div>
        <div className="border-t border-black pt-4 text-right">
          <p className="text-xs uppercase tracking-widest font-bold text-black/60">DR-SUGAR AI-Assisted Screening</p>
          <p className="text-[10px] mt-2 text-black/40">Research Prototype — Not for Clinical Use</p>
        </div>
      </div>

      <div className="mt-8 text-center print:hidden">
        <Link to="/reports" className="text-sm underline text-blue-700">← Back to Reports</Link>
      </div>
    </div>
  );
};

export default DoctorReport;
