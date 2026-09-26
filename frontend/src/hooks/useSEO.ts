import { useEffect } from 'react';

interface SEOProps {
  title?: string;
  description?: string;
}

export function useSEO({ title, description }: SEOProps) {
  useEffect(() => {
    if (title) {
      document.title = `${title} | DR-SUGAR`;
    }
    if (description) {
      const metaDescription = document.querySelector('meta[name="description"]');
      if (metaDescription) {
        metaDescription.setAttribute('content', description);
      }
    }
  }, [title, description]);
}
