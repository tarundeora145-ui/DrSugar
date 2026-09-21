import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

const API_BASE_URL = '/api';

const Dashboard = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE_URL}/dashboard`)
      .then(res => res.json())
      .then(resData => {
        if (resData.success) {
          setData(resData.data);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="py-24 space-y-16 animate-fade-in max-w-6xl mx-auto">
      <div>
        <h2 className="text-6xl font-light tracking-tighter mb-4 uppercase">Operational Dashboard</h2>
        <p className="text-xl text-muted-foreground font-light">Overview of screening activities and system metrics.</p>
      </div>
      
      {loading ? (
        <div className="text-muted-foreground uppercase tracking-widest text-sm animate-pulse">Loading Metrics...</div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            <div className="space-y-4 border border-muted p-8 bg-card/20">
              <h3 className="text-accent text-5xl font-light">{data?.totalScreenings || 0}</h3>
              <p className="text-muted-foreground uppercase tracking-widest text-sm font-bold">Total Screenings</p>
            </div>
            <div className="space-y-4 border border-muted p-8 bg-card/20">
              <h3 className="text-destructive text-5xl font-light">{data?.pendingReview || 0}</h3>
              <p className="text-muted-foreground uppercase tracking-widest text-sm font-bold">Pending Review</p>
            </div>
            <div className="space-y-4 border border-muted p-8 bg-card/20">
              <h3 className="text-emerald-500 text-5xl font-light">{data?.systemStatus || 'OFFLINE'}</h3>
              <p className="text-muted-foreground uppercase tracking-widest text-sm font-bold">System Status</p>
            </div>
          </div>

          <div className="mt-16 border-t border-muted pt-16">
            <h3 className="text-2xl font-light mb-8 uppercase tracking-widest">Recent Activity</h3>
            {data?.recentScreenings?.length === 0 ? (
              <p className="text-muted-foreground text-sm uppercase tracking-widest">No recent screenings.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-left">
                  <thead>
                    <tr className="border-b border-muted">
                      <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">ID</th>
                      <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Date</th>
                      <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Patient</th>
                      <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Grade</th>
                      <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Status</th>
                      <th className="p-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data?.recentScreenings?.map((screening: any) => (
                      <tr key={screening.id} className="border-b border-muted hover:bg-muted/10 transition-colors">
                        <td className="p-4 font-mono text-sm">{screening.display_id || screening.id}</td>
                        <td className="p-4 text-sm">{new Date(screening.created_at).toLocaleDateString('en-GB', { timeZone: 'Asia/Kolkata' })}</td>
                        <td className="p-4 font-bold">{screening.patient_name || 'N/A'}</td>
                        <td className="p-4">Grade {screening.result_grade ?? '-'}</td>
                        <td className="p-4 text-xs font-bold uppercase tracking-widest">
                          {screening.referable_dr ? <span className="text-destructive">Refer</span> : <span className="text-emerald-400">Routine</span>}
                        </td>
                        <td className="p-4">
                          <Link to={`/reports/${screening.id}/doctor`} className="text-sm uppercase tracking-widest hover:text-primary transition-colors">
                            View
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;
