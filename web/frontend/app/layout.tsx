import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';
import AnimatedBackground from '@/components/AnimatedBackground';

export const metadata: Metadata = {
  title: 'UNLET-ADAS',
  description:
    'Real-time low-light enhancement and detection for intelligent vehicle systems.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="relative min-h-screen overflow-x-hidden bg-base text-textMain font-sans antialiased">
        <AnimatedBackground />
        <div className="relative z-10">
          <Navbar />
          <main className="mx-auto max-w-6xl px-4 pb-16 pt-6 sm:px-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
