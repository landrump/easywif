import Image from 'next/image'

export default function MobileCompatibility() {
  return (
    <section className="py-16 md:py-24 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          {/* Left: Text Content (Desktop) / Above Image (Mobile) */}
          <div className="order-1 md:order-1 space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold text-navy">
              Plans don't only change at your desk
            </h2>
            <p className="text-lg text-navy-light leading-relaxed">
              Inspiration, decisions, and questions don't wait for the next planning meeting.
              EasyWIF is designed to be readable and insightful across devices — so you can review scenarios, understand impact, and stay aligned wherever you are.
            </p>
            <p className="text-sm text-navy-light">
              Perfect for quick reviews, discussions, and decision prep on the go.
            </p>
          </div>

          {/* Right: Image (Desktop) / Below Text (Mobile) */}
          <div className="order-2 md:order-2">
            <Image
              src="/EasyWIF Mobile View.png"
              alt="EasyWIF mobile view"
              width={600}
              height={800}
              className="rounded-lg shadow-xl w-full h-auto"
            />
          </div>
        </div>
      </div>
    </section>
  )
}

