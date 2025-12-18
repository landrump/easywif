import Image from 'next/image'

export default function Hero() {
  return (
    <section className="py-16 md:py-24 lg:py-32 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          {/* Left Side - Copy */}
          <div className="space-y-6">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-navy leading-tight">
              Scenario planning in minutes — not weeks.
            </h1>
            
            <p className="text-lg md:text-xl text-navy-light leading-relaxed">
              EasyWIF helps teams model planning changes quickly: filter your forecast, apply what-if scenarios, see the impact instantly, and export updated plans.
            </p>

            {/* Bullet Benefits */}
            <ul className="space-y-3 pt-4">
              <li className="flex items-start">
                <svg className="w-6 h-6 text-primary mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-navy-light">Answer what-if questions fast</span>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-primary mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-navy-light">See before vs after impact instantly</span>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-primary mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-navy-light">Export updated data back into your workflow</span>
              </li>
            </ul>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row gap-4 pt-6">
              <a
                href="#book-demo"
                className="bg-primary text-white px-8 py-4 rounded-lg hover:bg-primary-dark transition-colors font-semibold text-center"
              >
                Book a Personalized Demo
              </a>
              <a
                href="#video-overview"
                className="border-2 border-primary text-primary px-8 py-4 rounded-lg hover:bg-primary hover:text-white transition-colors font-semibold text-center"
              >
                Watch Overview
              </a>
            </div>
          </div>

          {/* Right Side - Visual */}
          <div className="relative overflow-hidden rounded-lg shadow-xl" style={{ height: '500px' }}>
            <div style={{ height: '100%', overflow: 'hidden' }}>
              <Image
                src="/EasyWIF_Hero.png"
                alt="EasyWIF Dashboard Screenshot"
                width={800}
                height={600}
                className="w-full h-full object-cover object-top"
                priority
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

