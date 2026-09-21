/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Don't auto-generate AGENTS.md/CLAUDE.md in this directory.
  agentRules: false,
};

export default nextConfig;
