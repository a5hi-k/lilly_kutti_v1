import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      '/avatar-videos': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/lipsync-tmp': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ref-assets': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] =
              'no-store, no-cache, must-revalidate, max-age=0'
          })
        },
      },
      '/generated-costumes': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] =
              'no-store, no-cache, must-revalidate, max-age=0'
          })
        },
      },
      '/payment-qr': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/session-uploads': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
