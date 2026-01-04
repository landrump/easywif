export default function EarlyAccess() {
  const features = [
    {
      title: 'Personalized walkthrough',
      description: 'Get a personalized demo that shows how EasyWIF works with your specific planning needs.',
    },
    {
      title: 'Demo with simulated data',
      description: 'We'll use simulated data or a template that matches your format to show how EasyWIF works with your planning needs.',
    },
    {
      title: 'Fast feedback loop',
      description: 'Share what you need, and we\'ll iterate quickly to make EasyWIF work for your workflow.',
    },
  ]

  return (
    <section className="py-16 md:py-24 bg-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-navy mb-4">
            Early access, guided demos.
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-12">
          {features.map((feature, index) => (
            <div key={index} className="text-center">
              <h3 className="text-xl font-semibold text-navy mb-3">
                {feature.title}
              </h3>
              <p className="text-navy-light">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

