const isStaticExport = process.env.NEXT_STATIC_EXPORT === '1';

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Don't auto-generate AGENTS.md/CLAUDE.md in this directory.
  agentRules: false,
  allowedDevOrigins: ['*.trycloudflare.com', 'localhost', '127.0.0.1'],
  ...(isStaticExport
    ? {
        output: 'export',
        distDir: 'out',
        images: { unoptimized: true },
      }
    : {
        async rewrites() {
          const backend =
            process.env.BACKEND_INTERNAL_URL ||
            process.env.NEXT_PUBLIC_API_URL ||
            'http://127.0.0.1:8000';
          return [
            {
              source: '/api/:path*',
              destination: `${backend.replace(/\/$/, '')}/api/:path*`,
            },
          ];
        },
      }),
};

export default nextConfig;
