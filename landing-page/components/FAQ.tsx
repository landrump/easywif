'use client'

import { useState } from 'react'

interface FAQItem {
  question: string
  answer: string
}

const faqs: FAQItem[] = [
  {
    question: 'Is this for small businesses or larger teams?',
    answer: 'EasyWIF works for teams of any size that need to model planning scenarios. We\'ve designed it to be simple enough for small teams but powerful enough for larger organizations that need faster scenario planning without enterprise complexity.',
  },
  {
    question: 'Do I need to change how my data is structured?',
    answer: 'Not necessarily. EasyWIF can work with common planning formats. During your demo, we can show you how your data maps to EasyWIF, or we can use a template that matches your structure.',
  },
  {
    question: 'Can I export results back into my spreadsheet or system?',
    answer: 'Yes. EasyWIF is designed to fit into your existing workflow. You can export updated plans in common formats to bring the results back into your spreadsheets or other planning tools.',
  },
  {
    question: 'Is EasyWIF AI-powered?',
    answer: 'EasyWIF focuses on fast scenario control and clear impact visualization. Optional AI insights may come later — today the value is fast scenario control without the complexity.',
  },
  {
    question: 'How do demos work?',
    answer: 'Book a demo and we\'ll schedule a personalized walkthrough. You can use your actual data or we can work with a template that matches your format. The demo is founder-led, so you\'ll get direct answers to your questions.',
  },
  {
    question: 'What does early access mean?',
    answer: 'Early access means you get guided onboarding, direct feedback channels, and the ability to shape EasyWIF as we build. We\'re focused on making it work for real planning teams, not just shipping features.',
  },
]

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(null)

  const toggleFAQ = (index: number) => {
    setOpenIndex(openIndex === index ? null : index)
  }

  return (
    <section id="faq" className="py-16 md:py-24 bg-gray-100 scroll-mt-16">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-navy mb-4">
            Frequently Asked Questions
          </h2>
        </div>

        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="bg-white rounded-lg border border-gray-200 overflow-hidden"
            >
              <button
                onClick={() => toggleFAQ(index)}
                className="w-full px-6 py-4 text-left flex justify-between items-center hover:bg-gray-50 transition-colors"
                aria-expanded={openIndex === index}
              >
                <span className="font-semibold text-navy pr-4">
                  {faq.question}
                </span>
                <svg
                  className={`w-5 h-5 text-primary flex-shrink-0 transition-transform ${
                    openIndex === index ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>
              {openIndex === index && (
                <div className="px-6 pb-4">
                  <p className="text-navy-light leading-relaxed">
                    {faq.answer}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

