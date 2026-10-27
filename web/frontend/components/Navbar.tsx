'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';

const links = [
  { href: '/', label: 'Image', icon: '🖼️' },
  { href: '/video', label: 'Video', icon: '🎬' },
  { href: '/live', label: 'Live', icon: '📡' },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="glass sticky top-0 z-30 border-b border-border/80">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <motion.div
          className="flex items-center gap-3"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
        >
          <motion.div
            whileHover={{ rotate: 8, scale: 1.08 }}
            transition={{ type: 'spring', stiffness: 300, damping: 15 }}
            className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-accent-gradient text-sm font-extrabold text-base shadow-glow"
          >
            UA
            <span className="absolute -right-0.5 -top-0.5 flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-ok opacity-75" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-ok" />
            </span>
          </motion.div>
          <div className="leading-tight">
            <div className="flex items-center gap-2">
              <span className="text-sm font-extrabold tracking-tight text-textMain">
                UNLET-ADAS
              </span>
              <span className="hidden rounded-full border border-accent/30 bg-accent/10 px-2 py-0.5 font-mono text-[10px] font-bold text-accent sm:inline-block">
                DCE-Net + YOLOv8s
              </span>
            </div>
            <div className="text-[11px] text-textDim">
              Real-time Low-Light Enhancement &amp; Perception HUD
            </div>
          </div>
        </motion.div>

        <nav className="relative flex items-center gap-1 rounded-full border border-border bg-card/90 p-1 shadow-inner">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className="relative rounded-full px-4 py-1.5 text-sm font-semibold transition-colors"
              >
                {active && (
                  <motion.span
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-full bg-accent-gradient shadow-glow"
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                  />
                )}
                <span
                  className={`relative z-10 flex items-center gap-1.5 ${
                    active ? 'text-base font-bold' : 'text-textDim hover:text-textMain'
                  }`}
                >
                  <span className="hidden sm:inline">{l.icon}</span>
                  {l.label}
                </span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
