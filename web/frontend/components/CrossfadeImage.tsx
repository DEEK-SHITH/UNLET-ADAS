'use client';

import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

/**
 * Swaps images with a smooth crossfade instead of a hard `src` cut.
 * Decodes each incoming frame off-DOM first (img.decode()) so the
 * browser never paints a half-downloaded/partial frame — that decode
 * stall, not the crossfade itself, is what made the live stream look
 * choppy. A stale-response guard (seq ref) drops any frame that
 * finishes decoding after a newer one already has.
 */
export default function CrossfadeImage({
  src,
  alt,
  className,
}: {
  src: string | undefined;
  alt: string;
  className?: string;
}) {
  const [ready, setReady] = useState<string | null>(null);
  const seq = useRef(0);

  useEffect(() => {
    if (!src) return;
    const mySeq = ++seq.current;
    const img = new window.Image();
    img.src = src;
    const finish = () => {
      if (seq.current === mySeq) setReady(src);
    };
    img.decode().then(finish).catch(finish);
  }, [src]);

  return (
    <div className={`relative overflow-hidden ${className ?? ''}`}>
      <AnimatePresence mode="sync">
        {ready && (
          <motion.img
            key={ready}
            src={ready}
            alt={alt}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16, ease: 'easeOut' }}
            className="absolute inset-0 h-full w-full object-contain"
          />
        )}
      </AnimatePresence>
    </div>
  );
}
