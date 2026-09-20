import { useEffect, useRef } from 'react';

export const RetinalConstellation = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    // Particle settings
    const particleCount = 120;
    const maxDistance = 150;
    
    interface Particle {
      x: number;
      y: number;
      vx: number;
      vy: number;
      radius: number;
      color: string;
      baseColor: string;
    }

    const colors = [
      '#8B5CF6', // Violet
      '#F59E0B', // Amber
      '#14B8A6', // Teal
      '#6366F1', // Indigo/Blue-violet
      '#94A3B8'  // Muted silver/slate for depth
    ];

    const particles: Particle[] = [];
    
    for (let i = 0; i < particleCount; i++) {
      const baseColor = colors[Math.floor(Math.random() * colors.length)];
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.8,
        radius: Math.random() * 2 + 0.5,
        color: baseColor,
        baseColor: baseColor
      });
    }

    let mouseX = -1000;
    let mouseY = -1000;

    const handleMouseMove = (e: MouseEvent) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    };
    window.addEventListener('mousemove', handleMouseMove);

    let animationFrameId: number;

    const animate = () => {
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      
      if (isLight) {
        ctx.fillStyle = 'rgba(248, 250, 252, 1)';
      } else {
        ctx.fillStyle = 'rgba(0, 0, 0, 1)';
      }
      ctx.fillRect(0, 0, width, height);

      // Update & Draw Particles
      for (let i = 0; i < particleCount; i++) {
        const p = particles[i];

        // Interaction with mouse
        const dxMouse = mouseX - p.x;
        const dyMouse = mouseY - p.y;
        const distMouse = Math.sqrt(dxMouse * dxMouse + dyMouse * dyMouse);
        
        if (distMouse < 200) {
           const force = (200 - distMouse) / 200;
           p.vx -= (dxMouse / distMouse) * force * 0.05;
           p.vy -= (dyMouse / distMouse) * force * 0.05;
        }

        // Apply friction
        p.vx *= 0.99;
        p.vy *= 0.99;

        // Base movement
        p.x += p.vx + (Math.random() - 0.5) * 0.2;
        p.y += p.vy + (Math.random() - 0.5) * 0.2;

        // Bounds
        if (p.x < 0 || p.x > width) p.vx *= -1;
        if (p.y < 0 || p.y > height) p.vy *= -1;
        p.x = Math.max(0, Math.min(width, p.x));
        p.y = Math.max(0, Math.min(height, p.y));

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();

        // Connect particles (vascular branching)
        for (let j = i + 1; j < particleCount; j++) {
          const p2 = particles[j];
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < maxDistance) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            
            // Opacity based on distance
            const alpha = 1 - (dist / maxDistance);
            
            // Color interpolation (simple blending)
            ctx.strokeStyle = isLight
              ? `rgba(124, 58, 237, ${alpha * 0.25})`
              : `rgba(139, 92, 246, ${alpha * 0.3})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      className="fixed inset-0 w-full h-full pointer-events-none z-[-1]"
    />
  );
};
