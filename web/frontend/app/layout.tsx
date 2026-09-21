import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';

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
      <body className="min-h-screen bg-base text-textMain font-sans antialiased">
        <Navbar />
        <main className="mx-auto max-w-6xl px-4 pb-16 pt-6 sm:px-6">
          {children}
        </main>
      </body>
    </html>
  );
}
