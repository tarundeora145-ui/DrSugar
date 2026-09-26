import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Introduction from './pages/Introduction';
import Dashboard from './pages/Dashboard';
import Screening from './pages/Screening';
import Reports from './pages/Reports';
import DoctorReport from './pages/DoctorReport';
import PatientReport from './pages/PatientReport';
import Simulation from './pages/Simulation';
import Validation from './pages/Validation';
import PrivacyPolicy from './pages/PrivacyPolicy';
import TermsConditions from './pages/TermsConditions';
import NotFound from './pages/NotFound';
import { cn } from './lib/utils';
import { ThemeProvider, useTheme } from './ThemeContext';

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

const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();
  return (
    <button
      onClick={toggleTheme}
      type="button"
      className="text-xs uppercase tracking-widest px-3 py-1.5 border border-border text-muted-foreground hover:text-foreground hover:border-foreground transition-colors cursor-pointer"
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      {theme === 'dark' ? 'LIGHT MODE' : 'DARK MODE'}
    </button>
  );
};

const Layout = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen bg-background text-foreground flex flex-col transition-colors duration-200">
    <header className="px-8 py-12 flex flex-col md:flex-row justify-between items-start md:items-center">
      <Link to="/" className="text-3xl font-light tracking-tighter text-foreground hover:text-primary transition-colors">
        DR-SUGAR
      </Link>
      <nav className="flex flex-wrap items-center gap-8 mt-8 md:mt-0">
        <NavLink to="/dashboard">Dashboard</NavLink>
        <NavLink to="/screening">Screening</NavLink>
        <NavLink to="/reports">Reports</NavLink>
        <NavLink to="/simulation">Simulation</NavLink>
        <NavLink to="/validation">Validation</NavLink>
        <ThemeToggle />
      </nav>
    </header>
    <main className="flex-1 px-8 max-w-7xl w-full mx-auto">
      {children}
    </main>
    <footer className="px-8 py-12 text-muted-foreground text-sm font-light mt-auto">
      <div className="border-t border-border pt-8 flex flex-col md:flex-row justify-between gap-4">
        <span>Explainable AI for Diabetic Retinopathy</span>
        <div className="flex gap-4">
          <Link to="/privacy" className="hover:text-foreground transition-colors">Privacy Policy</Link>
          <Link to="/terms" className="hover:text-foreground transition-colors">Terms & Conditions</Link>
        </div>
        <span>Prototype Version</span>
      </div>
    </footer>
  </div>
);

function App() {
  return (
    <ThemeProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Introduction />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/screening" element={<Screening />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/reports/:id/doctor" element={<DoctorReport />} />
            <Route path="/reports/:id/patient" element={<PatientReport />} />
            <Route path="/simulation" element={<Simulation />} />
            <Route path="/validation" element={<Validation />} />
            <Route path="/privacy" element={<PrivacyPolicy />} />
            <Route path="/terms" element={<TermsConditions />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
}

export default App;

