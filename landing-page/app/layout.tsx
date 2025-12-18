import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'EasyWIF - Scenario Planning in Minutes, Not Weeks',
  description: 'EasyWIF helps teams model planning changes quickly: filter your forecast, apply what-if scenarios, see the impact instantly, and export updated plans.',
  openGraph: {
    title: 'EasyWIF - Scenario Planning in Minutes, Not Weeks',
    description: 'EasyWIF helps teams model planning changes quickly: filter your forecast, apply what-if scenarios, see the impact instantly, and export updated plans.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'EasyWIF - Scenario Planning in Minutes, Not Weeks',
    description: 'EasyWIF helps teams model planning changes quickly: filter your forecast, apply what-if scenarios, see the impact instantly, and export updated plans.',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}

