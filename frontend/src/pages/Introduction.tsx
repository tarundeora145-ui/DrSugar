import { Link } from 'react-router-dom';
import { RetinalConstellation } from '../components/RetinalConstellation';

const Introduction = () => {
  return (
    <div className="relative min-h-screen w-full overflow-x-hidden text-white font-light">
      {/* Background Canvas */}
      <RetinalConstellation />

      {/* 1. Hero Section */}
      <section className="min-h-[85vh] flex flex-col justify-center px-8 max-w-7xl mx-auto pt-20">
        <h1 className="text-7xl md:text-9xl font-light tracking-tighter mb-4 text-white">DR-SUGAR</h1>
        <h2 className="text-2xl md:text-4xl text-primary font-light mb-8 max-w-3xl leading-tight">
          Explainable AI for Diabetic Retinopathy Screening
        </h2>
        
        <p className="text-xl md:text-3xl text-muted-foreground font-light mb-16 max-w-2xl leading-relaxed">
          Understand the image.<br/>
          Understand the evidence.<br/>
          <span className="text-white">Support the decision.</span>
        </p>

        <div className="flex flex-col sm:flex-row gap-6">
          <Link 
            to="/screening" 
            className="px-10 py-5 border border-primary text-primary hover:bg-primary/10 transition-colors uppercase tracking-widest text-sm text-center"
          >
            Start Screening
          </Link>
          <a 
            href="#how-it-works"
            className="px-10 py-5 border border-muted text-muted-foreground hover:border-white hover:text-white transition-colors uppercase tracking-widest text-sm text-center"
          >
            Explore the Pipeline
          </a>
        </div>
      </section>

      {/* 2. The Screening Problem */}
      <section className="py-32 px-8 max-w-7xl mx-auto bg-black/80 backdrop-blur-md border-t border-muted">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-16">
          <div>
            <h3 className="text-4xl font-light tracking-tighter mb-8">THE SCALING LIMIT</h3>
          </div>
          <div className="space-y-8 text-lg text-muted-foreground leading-relaxed">
            <p>
              Rural healthcare networks face a massive volume of diabetic patients requiring preventative screening, with fewer ophthalmologists available to review them. Traditional AI promises to filter these queues, but introduces a new problem: <span className="text-white font-medium">the black box</span>.
            </p>
            <p>
              A single probabilistic score is insufficient for clinical trust. Doctors cannot refer patients based solely on uncalibrated confidence percentages without understanding the anatomical reasoning behind the network's choice.
            </p>
          </div>
        </div>
      </section>

      {/* 3. How DR-SUGAR Works */}
      <section id="how-it-works" className="py-32 px-8 max-w-7xl mx-auto bg-black/80 backdrop-blur-md border-t border-muted">
        <h3 className="text-4xl font-light tracking-tighter mb-24 text-center">THE PIPELINE</h3>
        
        <div className="flex flex-col space-y-24 max-w-4xl mx-auto text-center">
          <div>
             <span className="block text-primary text-sm uppercase tracking-widest mb-4">Stage 01</span>
             <h4 className="text-3xl font-light text-white mb-4">Quality Gate</h4>
             <p className="text-muted-foreground">Assessing illumination, focus, and field-of-view before allowing diagnostic inference.</p>
          </div>
          <div className="w-[1px] h-16 bg-muted mx-auto"></div>
          <div>
             <span className="block text-primary text-sm uppercase tracking-widest mb-4">Stage 02</span>
             <h4 className="text-3xl font-light text-white mb-4">Retinal Analysis</h4>
             <p className="text-muted-foreground">Isolating structural landmarks including the optic disc, fovea, and detailed vascular networks.</p>
          </div>
          <div className="w-[1px] h-16 bg-muted mx-auto"></div>
          <div>
             <span className="block text-primary text-sm uppercase tracking-widest mb-4">Stage 03</span>
             <h4 className="text-3xl font-light text-white mb-4">Lesion Detection</h4>
             <p className="text-muted-foreground">Mapping microaneurysms, hemorrhages, and exudates directly to the grading logic.</p>
          </div>
          <div className="w-[1px] h-16 bg-muted mx-auto"></div>
          <div>
             <span className="block text-accent text-sm uppercase tracking-widest mb-4">Stage 04</span>
             <h4 className="text-3xl font-light text-white mb-4">Explainable Grading</h4>
             <p className="text-muted-foreground">Merging structural evidence with Grad-CAM overlays to produce a clinically auditable DR Grade.</p>
          </div>
        </div>
      </section>

      {/* 4. Why Explainability Matters */}
      <section className="py-32 px-8 max-w-7xl mx-auto bg-black/80 backdrop-blur-md border-t border-muted">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
          <div className="order-2 md:order-1 border border-muted aspect-square flex flex-col items-center justify-center bg-card/20 p-8 text-center">
            <span className="text-4xl text-amber-500 mb-4">⚠️</span>
            <h5 className="text-xl font-light text-white mb-2">Black Box Inference</h5>
            <p className="text-sm text-muted-foreground uppercase tracking-widest">Uncalibrated / Unavailable</p>
          </div>
          <div className="order-1 md:order-2 space-y-8 text-lg text-muted-foreground leading-relaxed">
            <h3 className="text-4xl font-light tracking-tighter text-white mb-8">RADICAL TRANSPARENCY</h3>
            <p>
              Explainability is not a decorative heatmap. It is the core requirement for medical deployment. 
            </p>
            <p>
              DR-SUGAR strictly isolates inference from decoration. If the model cannot provide structural evidence or explicitly highlight the lesions driving its decision, the system formally rejects the confidence score. No pseudo-metrics. No hallucinations.
            </p>
          </div>
        </div>
      </section>

      {/* 5. Multi-Dataset Intelligence */}
      <section className="py-32 px-8 max-w-7xl mx-auto bg-black/80 backdrop-blur-md border-t border-muted">
         <h3 className="text-4xl font-light tracking-tighter mb-16 text-center">CLINICAL VALIDATION</h3>
         <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="border border-muted p-8 text-center">
              <h4 className="text-2xl font-light text-white mb-2">APTOS</h4>
              <p className="text-sm text-muted-foreground uppercase tracking-widest">Blindness Detection</p>
            </div>
            <div className="border border-muted p-8 text-center">
              <h4 className="text-2xl font-light text-white mb-2">IDRiD</h4>
              <p className="text-sm text-muted-foreground uppercase tracking-widest">Indian Retinopathy</p>
            </div>
            <div className="border border-muted p-8 text-center">
              <h4 className="text-2xl font-light text-white mb-2">DRIVE</h4>
              <p className="text-sm text-muted-foreground uppercase tracking-widest">Vessel Segmentation</p>
            </div>
            <div className="border border-muted p-8 text-center">
              <h4 className="text-2xl font-light text-white mb-2">MESSIDOR</h4>
              <p className="text-sm text-muted-foreground uppercase tracking-widest">Robustness</p>
            </div>
         </div>
      </section>

      {/* 6. Final CTA */}
      <section className="py-32 px-8 max-w-7xl mx-auto bg-black border-t border-muted flex flex-col items-center justify-center text-center">
        <h2 className="text-5xl md:text-7xl font-light tracking-tighter mb-16">EVALUATE THE PIPELINE</h2>
        <Link 
            to="/screening" 
            className="px-12 py-6 border border-primary text-primary hover:bg-primary/10 transition-colors uppercase tracking-widest text-lg"
          >
            Start Screening
        </Link>
      </section>
      
    </div>
  );
};

export default Introduction;
