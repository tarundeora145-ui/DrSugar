
import { useSEO } from '../hooks/useSEO';

export default function PrivacyPolicy() {
  useSEO({ title: 'Privacy Policy', description: 'Privacy policy for DR-SUGAR screening.' });

  return (
    <div className="py-12 max-w-3xl space-y-8">
      <h1 className="text-4xl font-light tracking-tighter">Privacy Policy</h1>
      <p className="text-muted-foreground">Last updated: {new Date().toLocaleDateString()}</p>
      
      <section className="space-y-4">
        <h2 className="text-xl font-medium tracking-wide">Data Collection</h2>
        <p className="text-muted-foreground leading-relaxed">
          DR-SUGAR collects only the minimum necessary patient metadata and fundus images required to perform diabetic retinopathy screening. 
          All processing is performed with privacy in mind.
        </p>
      </section>
      
      <section className="space-y-4">
        <h2 className="text-xl font-medium tracking-wide">Data Usage</h2>
        <p className="text-muted-foreground leading-relaxed">
          Medical images and derived diagnostic metadata are strictly used to compute the DR severity and generate clinical reports. 
          We do not sell data to third parties.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-medium tracking-wide">Contact</h2>
        <p className="text-muted-foreground leading-relaxed">
          For any privacy concerns, please contact our medical informatics team.
        </p>
      </section>
    </div>
  );
}
