
const Dashboard = () => {
  return (
    <div className="py-24 space-y-16">
      <div>
        <h2>Operational Dashboard</h2>
        <p className="text-muted-foreground mt-4">Overview of screening activities and system metrics.</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
        <div className="space-y-4">
          <h3 className="text-accent text-5xl">--</h3>
          <p className="text-muted-foreground uppercase tracking-widest text-sm">Total Screenings</p>
        </div>
        <div className="space-y-4">
          <h3 className="text-primary text-5xl">--</h3>
          <p className="text-muted-foreground uppercase tracking-widest text-sm">Pending Review</p>
        </div>
        <div className="space-y-4">
          <h3 className="text-success text-5xl">--</h3>
          <p className="text-muted-foreground uppercase tracking-widest text-sm">System Status</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
