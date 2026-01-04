import Image from 'next/image'

export default function Founder() {
  return (
    <section className="py-16 md:py-24 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-2 gap-12 items-start">
          {/* Main text - appears first on mobile, right side on desktop */}
          <div className="order-1 md:order-2 space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold text-navy">
              Built by someone who's lived the problem
            </h2>
            <p className="text-lg text-navy-light leading-relaxed">
              EasyWIF was created by an 18-year technology and operations leader who has spent years managing forecasts, roadmaps, and resource plans across real teams and real constraints.
            </p>
            <p className="text-lg text-navy-light leading-relaxed">
              It's designed to solve everyday planning friction — not add more process, complexity, or overhead.
            </p>
            
            {/* Attribution - desktop only, appears in text column */}
            <div className="pt-4 space-y-2 hidden md:block">
              <p className="font-semibold text-navy">
                Paul Landrum — Founder, EasyWIF
              </p>
              <p className="text-sm text-navy-light">
                18+ years in technology, finance, and operations
              </p>
              <p className="text-sm">
                <a 
                  href="https://www.linkedin.com/in/paullandrum" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  LinkedIn: www.linkedin.com/in/paullandrum
                </a>
              </p>
            </div>
          </div>

          {/* Founder Image - appears second on mobile, left side on desktop */}
          <div className="order-2 md:order-1">
            <Image
              src="/Paul_EasyWIF Avatar.png"
              alt="Paul Landrum"
              width={500}
              height={600}
              className="rounded-lg shadow-lg w-full h-auto"
            />
          </div>
          
          {/* Attribution - mobile only, appears after image */}
          <div className="order-3 md:hidden pt-4 space-y-2">
            <p className="font-semibold text-navy">
              Paul Landrum — Founder, EasyWIF
            </p>
            <p className="text-sm text-navy-light">
              18+ years in technology, finance, and operations
            </p>
            <p className="text-sm">
              <a 
                href="https://www.linkedin.com/in/paullandrum" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                LinkedIn: www.linkedin.com/in/paullandrum
              </a>
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

