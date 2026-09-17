import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Introduction from './pages/Introduction';
import Dashboard from './pages/Dashboard';
import Screening from './pages/Screening';
import Reports from './pages/Reports';
import Simulation from './pages/Simulation';
import Validation from './pages/Validation';
import { cn } from './lib/utils';

const NavLink = ({ to, children }: { to: string, children: React.ReactNode }) => {
  const location = useLocation();
  const isActive = location.pathname === to;
  
  return (
    <Link 
      to={to} 
      className={cn(
        "transition-colors text-sm uppercase tracking-widest font-light",
        isActive ? "text-primary" : "text-muted-foreground hover:text-foreground"
      )}
    >
      {children}
    </Link>
  );
};

const Layout = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen bg-background text-foreground flex flex-col">
    <header className="px-8 py-12 flex flex-col md:flex-row justify-between items-start md:items-center">
      <Link to="/" className="text-3xl font-light tracking-tighter text-foreground hover:text-primary transition-colors">
        DR-SUGAR
      </Link>
      <nav className="flex flex-wrap gap-8 mt-8 md:mt-0">
        <NavLink to="/dashboard">Dashboard</NavLink>
        <NavLink to="/screening">Screening</NavLink>
        <NavLink to="/reports">Reports</NavLink>
        <NavLink to="/simulation">Simulation</NavLink>
        <NavLink to="/validation">Validation</NavLink>
      </nav>
    </header>
    <main className="flex-1 px-8 max-w-7xl w-full mx-auto">
      {children}
    </main>
    <footer className="px-8 py-12 text-muted-foreground text-sm font-light mt-auto">
      <div className="border-t border-muted pt-8 flex justify-between">
        <span>Explainable AI for Diabetic Retinopathy</span>
        <span>Prototype Version</span>
      </div>
    </footer>
  </div>
);

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Introduction />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/screening" element={<Screening />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/simulation" element={<Simulation />} />
          <Route path="/validation" element={<Validation />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
