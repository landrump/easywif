export default function UseCases() {
  const useCases = [
    {
      title: 'Post-production & media planning',
      description: 'Model schedule shifts, resource reallocations, and budget scenarios for production timelines.',
      bullets: ['Shift schedules', 'Scale teams', 'Export plan'],
    },
    {
      title: 'Consulting & delivery capacity planning',
      description: 'Quickly adjust project assignments, model new engagements, and see capacity impact.',
      bullets: ['Shift schedules', 'Scale teams', 'Export plan'],
    },
    {
      title: 'Construction project planning',
      description: 'Plan timeline changes, resource adjustments, and cost scenarios across multiple projects.',
      bullets: ['Shift schedules', 'Scale teams', 'Export plan'],
    },
    {
      title: 'Operations & workforce planning',
      description: 'Model staffing changes, shift adjustments, and capacity scenarios for operations teams.',
      bullets: ['Shift schedules', 'Scale teams', 'Export plan'],
    },
    {
      title: 'Finance planning & budgeting support',
      description: 'Run budget scenarios, model cost changes, and see financial impact before committing.',
      bullets: ['Shift schedules', 'Scale teams', 'Export plan'],
    },
    {
      title: 'Portfolio / roadmap scenario reviews',
      description: 'Compare roadmap options, model priority shifts, and understand trade-offs quickly.',
      bullets: ['Shift schedules', 'Scale teams', 'Export plan'],
    },
  ]

  return (
    <section id="use-cases" className="py-16 md:py-24 bg-white scroll-mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-navy mb-4">
            Works anywhere plans change.
          </h2>
          <p className="text-xl text-navy-light max-w-3xl mx-auto">
            If your business uses forecasts, capacity plans, or project roadmaps — EasyWIF helps you model change quickly.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {useCases.map((useCase, index) => (
            <div key={index} className="bg-gray-50 p-6 rounded-lg border border-gray-200 hover:shadow-md transition-shadow">
              <h3 className="text-lg font-semibold text-navy mb-2">
                {useCase.title}
              </h3>
              <p className="text-navy-light text-sm mb-4">
                {useCase.description}
              </p>
              <ul className="space-y-1">
                {useCase.bullets.map((bullet, i) => (
                  <li key={i} className="text-sm text-navy-light flex items-center">
                    <span className="w-1.5 h-1.5 bg-primary rounded-full mr-2"></span>
                    {bullet}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <p className="text-center text-navy-light max-w-2xl mx-auto">
          EasyWIF is currently demo-led and best for teams that want faster scenario turns without enterprise overhead.
        </p>
      </div>
    </section>
  )
}

