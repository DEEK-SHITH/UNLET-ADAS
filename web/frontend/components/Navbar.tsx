'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const links = [
  { href: '/', label: 'Image' },
  { href: '/video', label: 'Video' },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="border-b border-border bg-panel/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-gradient text-sm font-bold text-base">
            UA
          </div>
          <div className="leading-tight">
            <div className="text-sm font-bold tracking-tight text-textMain">
              UNLET-ADAS
            </div>
            <div className="text-[11px] text-textDim">
              Low-light enhancement &amp; detection
            </div>
          </div>
        </div>

        <nav className="flex items-center gap-1 rounded-full border border-border bg-card p-1">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-full px-4 py-1.5 text-sm font-medium transition ${
                  active
                    ? 'bg-accent-gradient text-base'
                    : 'text-textDim hover:text-textMain'
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
