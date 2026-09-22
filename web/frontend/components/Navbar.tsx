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
    <header className="glass sticky top-0 z-20 border-b border-border">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <motion.div
          className="flex items-center gap-2.5"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
        >
          <motion.div
            whileHover={{ rotate: 8, scale: 1.08 }}
            transition={{ type: 'spring', stiffness: 300, damping: 15 }}
            className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-gradient text-sm font-bold text-base shadow-glow"
          >
            UA
          </motion.div>
          <div className="leading-tight">
            <div className="text-sm font-bold tracking-tight text-textMain">
              UNLET-ADAS
            </div>
            <div className="text-[11px] text-textDim">
              Low-light enhancement &amp; detection
            </div>
          </div>
        </motion.div>

        <nav className="relative flex items-center gap-1 rounded-full border border-border bg-card p-1">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className="relative rounded-full px-4 py-1.5 text-sm font-medium transition-colors"
              >
                {active && (
                  <motion.span
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-full bg-accent-gradient"
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                  />
                )}
                <span
                  className={`relative z-10 ${active ? 'text-base' : 'text-textDim hover:text-textMain'}`}
                >
                  <span className="mr-1.5 hidden sm:inline">{l.icon}</span>
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
