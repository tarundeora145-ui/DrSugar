
import { SimulationDashboard } from '../components/SimulationDashboard';

const Simulation = () => {
  return (
    <div className="py-24 space-y-16 animate-fade-in">
      <section className="max-w-2xl">
        <h1 className="text-6xl font-light tracking-tighter mb-6">DEPLOYMENT SIMULATION</h1>
        <p className="text-xl text-muted-foreground font-light leading-relaxed">
          Simulate throughput, bandwidth constraints, and clinical queueing for large-scale rural deployments.
        </p>
      </section>

      <section className="w-full">
        <SimulationDashboard />
      </section>
    </div>
  );
};

export default Simulation;
