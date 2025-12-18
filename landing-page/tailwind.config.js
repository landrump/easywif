/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0071C5',
          dark: '#005a9e',
        },
        accent: {
          DEFAULT: '#14b8a6',
          light: '#5eead4',
        },
        navy: {
          DEFAULT: '#1e293b',
          light: '#334155',
        }
      },
    },
  },
  plugins: [],
}

