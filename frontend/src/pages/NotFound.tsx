
import { Link } from 'react-router-dom';
import { useSEO } from '../hooks/useSEO';

export default function NotFound() {
  useSEO({ title: 'Page Not Found', description: 'The requested page could not be found.' });

  return (
    <div className="py-24 flex flex-col items-center justify-center space-y-8 text-center h-[50vh]">
      <h1 className="text-6xl font-light tracking-tighter text-muted-foreground">404</h1>
      <h2 className="text-2xl font-medium tracking-wide">Page Not Found</h2>
      <p className="text-muted-foreground max-w-md">
        The screening report or module you are looking for does not exist or has been moved.
      </p>
      <Link 
        to="/" 
        className="mt-8 px-6 py-3 border border-primary text-primary hover:bg-primary/10 transition-colors uppercase tracking-widest text-sm"
      >
        Return to Home
      </Link>
    </div>
  );
}
