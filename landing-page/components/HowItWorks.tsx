import Image from 'next/image'

export default function HowItWorks() {
  const steps = [
    {
      title: 'Filter Data',
      description: 'Quickly filter your forecast or plan by project, team, time period, or any dimension that matters.',
    },
    {
      title: 'Make a Change',
      description: 'Apply what-if scenarios: shift timelines, adjust resources, change priorities, or model new initiatives.',
    },
    {
      title: 'Review Impact',
      description: 'See before vs after comparisons instantly. Understand capacity, spend, and schedule implications at a glance.',
    },
  ]

  return (
    <section id="how-it-works" className="py-16 md:py-24 bg-white scroll-mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-navy mb-4">
            How EasyWIF Works
          </h2>
          <p className="text-xl text-navy-light max-w-2xl mx-auto">
            A simple workflow that turns planning changes into clear impact.
          </p>
        </div>

        {/* Main Image */}
        <div className="mb-12">
          <Image
            src="/EasyWIF_HowItWorks.jpg"
            alt="How EasyWIF Works - 3 Step Process"
            width={1200}
            height={600}
            className="rounded-lg shadow-lg w-full h-auto"
          />
        </div>

        {/* Step Cards */}
        <div className="grid md:grid-cols-3 gap-8">
          {steps.map((step, index) => (
            <div key={index} className="bg-gray-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold text-navy mb-3">
                {step.title}
              </h3>
              <p className="text-navy-light">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

