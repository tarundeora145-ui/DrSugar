import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

const API_BASE_URL = '/api';

const Reports = () => {
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE_URL}/reports`)
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setReports(data.data);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="py-24 max-w-6xl mx-auto animate-fade-in">
      <div className="mb-12">
        <h1 className="text-6xl font-light tracking-tighter mb-4">REPORTS</h1>
        <p className="text-xl text-muted-foreground font-light">
          View completed clinical screening reports and AI explainability records.
        </p>
      </div>

      {loading ? (
        <div className="text-muted-foreground uppercase tracking-widest text-sm animate-pulse">Loading Reports...</div>
      ) : reports.length === 0 ? (
        <div className="border border-muted p-12 text-center bg-card/30">
          <p className="text-muted-foreground text-lg font-light mb-4">No completed screenings found.</p>
          <Link to="/screening" className="text-sm underline">Start a new screening</Link>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-muted">
                <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">ID</th>
                <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Date</th>
                <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Patient</th>
                <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Result</th>
                <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Action</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((report) => (
                <tr key={report.id} className="border-b border-muted hover:bg-muted/10 transition-colors">
                  <td className="p-4 font-mono text-sm">{report.display_id || report.id}</td>
                  <td className="p-4 text-sm">{new Date(report.created_at).toLocaleDateString('en-GB', { timeZone: 'Asia/Kolkata' })}</td>
                  <td className="p-4">
                    <div className="font-bold">{report.patient_name || 'Unknown'}</div>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      <span className="font-bold">Grade {report.result_grade}</span>
                      {report.referable_dr ? (
                        <span className="px-2 py-0.5 text-[10px] uppercase tracking-widest bg-destructive/20 text-destructive font-bold">Refer</span>
                      ) : (
                        <span className="px-2 py-0.5 text-[10px] uppercase tracking-widest bg-emerald-900/30 text-emerald-400 font-bold">Routine</span>
                      )}
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="flex gap-4">
                      <Link to={`/reports/${report.id}/doctor`} className="text-sm uppercase tracking-widest hover:text-primary transition-colors">
                        Clinical
                      </Link>
                      <Link to={`/reports/${report.id}/patient`} className="text-sm uppercase tracking-widest hover:text-primary transition-colors">
                        Patient
                      </Link>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default Reports;
