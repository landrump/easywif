export default function UseCases() {
  const useCases = [
    {
      title: 'Post-production & media planning',
      bullets: ['Reschedule shoots', 'Reallocate editors', 'Update delivery timelines'],
      icon: 'video',
    },
    {
      title: 'Consulting & delivery capacity planning',
      bullets: ['Add new projects', 'Balance consultant workload', 'Identify capacity gaps'],
      icon: 'briefcase',
    },
    {
      title: 'Construction project planning',
      bullets: ['Shift project phases', 'Reallocate crews', 'Update cost forecasts'],
      icon: 'building',
    },
    {
      title: 'Operations & workforce planning',
      bullets: ['Adjust shift schedules', 'Scale up or down', 'Balance workload across teams'],
      icon: 'users',
    },
    {
      title: 'Finance planning & budgeting support',
      bullets: ['Model cost changes', 'Compare budget options', 'See impact on forecasts'],
      icon: 'chart',
    },
    {
      title: 'Portfolio / roadmap scenario reviews',
      bullets: ['Prioritize initiatives', 'Model trade-offs', 'Align resources to strategy'],
      icon: 'roadmap',
    },
  ]

  const getIcon = (iconType: string) => {
    const iconClass = "w-12 h-12 text-primary"
    switch (iconType) {
      case 'video':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        )
      case 'briefcase':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        )
      case 'building':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
          </svg>
        )
      case 'users':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
        )
      case 'chart':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        )
      case 'roadmap':
        return (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-1.447-.894L15 4m0 13V4m0 0L9 7" />
          </svg>
        )
      default:
        return null
    }
  }

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
            <div key={index} className="bg-gray-50 p-6 rounded-lg border border-gray-200 hover:shadow-md transition-shadow flex flex-col">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-semibold text-navy flex-1 pr-4">
                  {useCase.title}
                </h3>
                <div className="flex-shrink-0">
                  {getIcon(useCase.icon)}
                </div>
              </div>
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

