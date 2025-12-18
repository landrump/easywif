import Navigation from '@/components/Navigation'
import Hero from '@/components/Hero'
import Problem from '@/components/Problem'
import HowItWorks from '@/components/HowItWorks'
import Product from '@/components/Product'
import UseCases from '@/components/UseCases'
import EarlyAccess from '@/components/EarlyAccess'
import VideoSection from '@/components/VideoSection'
import FAQ from '@/components/FAQ'
import BookDemo from '@/components/BookDemo'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <main className="min-h-screen">
      <Navigation />
      <Hero />
      <Problem />
      <HowItWorks />
      <Product />
      <UseCases />
      <EarlyAccess />
      <VideoSection />
      <FAQ />
      <BookDemo />
      <Footer />
    </main>
  )
}

