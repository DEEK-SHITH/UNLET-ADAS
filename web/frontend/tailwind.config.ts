import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        base: '#07090f',
        panel: '#10141f',
        card: '#151b29',
        card2: '#1a2233',
        border: '#232c3f',
        borderLt: '#303c56',
        accent: '#5eead4',
        accent2: '#818cf8',
        textMain: '#eef1f8',
        textDim: '#8b96ad',
        danger: '#fb7185',
        warning: '#fbbf24',
        ok: '#4ade80',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(94,234,212,0.15), 0 8px 30px rgba(94,234,212,0.08)',
      },
      backgroundImage: {
        'accent-gradient': 'linear-gradient(135deg, #5eead4, #818cf8)',
      },
    },
  },
  plugins: [],
};

export default config;
