import Image from 'next/image'

export default function Product() {
  return (
    <section id="product" className="py-16 md:py-24 bg-gray-100 scroll-mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-navy mb-4">
            Built for clarity — not complexity.
          </h2>
          <p className="text-xl text-navy-light max-w-2xl mx-auto">
            A few powerful views to plan, compare, and decide.
          </p>
        </div>

        {/* Block 1: Project Planning View */}
        <div className="grid md:grid-cols-2 gap-12 items-center mb-20">
          <div className="order-2 md:order-1">
            <Image
              src="/EasyWIF_Project View.PNG"
              alt="EasyWIF Project Planning View"
              width={800}
              height={600}
              className="rounded-lg shadow-xl w-full h-auto"
            />
          </div>
          <div className="order-1 md:order-2 space-y-4">
            <h3 className="text-2xl md:text-3xl font-bold text-navy">
              Plan with timeline clarity
            </h3>
            <ul className="space-y-3">
              <li className="flex items-start">
                <svg className="w-6 h-6 text-primary mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-navy-light">See projects across months</span>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-primary mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-navy-light">Group by workstreams</span>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-primary mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-navy-light">Understand schedule implications instantly</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Block 2: Impact / Analytics View */}
        <div className="grid md:grid-cols-2 gap-12 items-center mb-20">
          <div className="space-y-4">
            <h3 className="text-2xl md:text-3xl font-bold text-navy">
              See the impact before you commit
            </h3>
            <ul className="space-y-3">
              <li className="flex items-start">
                <svg className="w-6 h-6 text-primary mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-navy-light">Compare current vs scenario</span>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-primary mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-navy-light">Spot capacity or spend pressure</span>
              </li>
              <li className="flex items-start">
                <svg className="w-6 h-6 text-primary mr-3 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-navy-light">Export updated outputs for decision cycles</span>
              </li>
            </ul>
          </div>
          <div>
            <Image
              src="/EasyWIF_Resource $K Impact.PNG"
              alt="EasyWIF Resource and Dollar Impact View"
              width={800}
              height={600}
              className="rounded-lg shadow-xl w-full h-auto"
            />
          </div>
        </div>

        {/* Optional: Save & Export Callout */}
        <div className="bg-white rounded-lg p-8 shadow-md">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <Image
                src="/EasyWIF_Save and Export_New.png"
                alt="EasyWIF Save and Export Feature"
                width={800}
                height={600}
                className="rounded-lg shadow-xl w-full h-auto"
              />
            </div>
            <div className="space-y-3">
              <h4 className="text-xl font-semibold text-navy">Save & Export</h4>
              <p className="text-navy-light">
                Save scenarios for presentations. Export updated data to avoid rework.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

