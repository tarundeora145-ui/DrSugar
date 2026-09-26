
import { useSEO } from '../hooks/useSEO';

export default function TermsConditions() {
  useSEO({ title: 'Terms & Conditions', description: 'Terms of use for DR-SUGAR screening prototype.' });

  return (
    <div className="py-12 max-w-3xl space-y-8">
      <h1 className="text-4xl font-light tracking-tighter">Terms & Conditions</h1>
      <p className="text-muted-foreground">Last updated: {new Date().toLocaleDateString()}</p>
      
      <section className="space-y-4">
        <h2 className="text-xl font-medium tracking-wide">Prototype Status</h2>
        <p className="text-muted-foreground leading-relaxed">
          DR-SUGAR is an Explainable AI research prototype for Diabetic Retinopathy screening. 
          It is not a final clinical diagnostic tool. All predictions require ophthalmologist review.
        </p>
      </section>
      
      <section className="space-y-4">
        <h2 className="text-xl font-medium tracking-wide">No Medical Advice</h2>
        <p className="text-muted-foreground leading-relaxed">
          The contents of the DR-SUGAR dashboard and generated reports do not constitute professional medical advice, diagnosis, or treatment. 
          Always seek the advice of your physician or other qualified health provider.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-medium tracking-wide">Liability</h2>
        <p className="text-muted-foreground leading-relaxed">
          The creators of DR-SUGAR shall not be held liable for any clinical decisions made based solely on the AI predictions.
        </p>
      </section>
    </div>
  );
}
