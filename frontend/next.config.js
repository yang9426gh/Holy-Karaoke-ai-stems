/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // We ship the frontend as a static bundle for the desktop app (served by the backend).
  output: 'export',
  images: { unoptimized: true },
}

module.exports = nextConfig
