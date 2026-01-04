'use client'

import { useState } from 'react'
import { calendlyLink, contactEmail } from '@/app/constants'

export default function BookDemo() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    notes: '',
  })
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // For Netlify Forms, add netlify attribute to form
    // This is a placeholder - you can use Netlify Forms or Calendly
    console.log('Form submitted:', formData)
    setSubmitted(true)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  return (
    <section id="book-demo" className="py-16 md:py-24 bg-white scroll-mt-16">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-navy mb-4">
            Want to see it with your workflow?
          </h2>
          <p className="text-xl text-navy-light">
            Book a demo and we'll walk through EasyWIF using simulated data or a quick template close to your format.
          </p>
        </div>

        {submitted ? (
          <div className="bg-accent-light text-white p-8 rounded-lg text-center">
            <h3 className="text-2xl font-semibold mb-2">Thank you!</h3>
            <p>We'll be in touch soon to schedule your demo.</p>
          </div>
        ) : (
          <div className="grid md:grid-cols-2 gap-8">
            {/* Primary CTA - Calendly or Direct Link */}
            <div className="bg-primary text-white p-8 rounded-lg text-center flex flex-col justify-center">
              <h3 className="text-2xl font-semibold mb-4">Book a Personalized Demo</h3>
              <p className="mb-6 opacity-90">
                Schedule a time that works for you
              </p>
              <a
                href={calendlyLink}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-white text-primary px-6 py-3 rounded-lg hover:bg-gray-100 transition-colors font-semibold inline-block"
              >
                Open Calendar
              </a>
              <p className="text-sm mt-4 opacity-75">
                Or use the form to request a time
              </p>
            </div>

            {/* Form Alternative */}
            <form
              name="demo-request"
              method="POST"
              data-netlify="true"
              onSubmit={handleSubmit}
              className="space-y-4"
            >
              <input type="hidden" name="form-name" value="demo-request" />
              
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-navy mb-1">
                  Name *
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  required
                  value={formData.name}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-navy mb-1">
                  Email *
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  required
                  value={formData.email}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                />
              </div>

              <div>
                <label htmlFor="company" className="block text-sm font-medium text-navy mb-1">
                  Company
                </label>
                <input
                  type="text"
                  id="company"
                  name="company"
                  value={formData.company}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                />
              </div>

              <div>
                <label htmlFor="notes" className="block text-sm font-medium text-navy mb-1">
                  Notes (optional)
                </label>
                <textarea
                  id="notes"
                  name="notes"
                  rows={3}
                  value={formData.notes}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                />
              </div>

              <button
                type="submit"
                className="w-full bg-primary text-white px-6 py-3 rounded-lg hover:bg-primary-dark transition-colors font-semibold"
              >
                Request Demo
              </button>

              <p className="text-sm text-navy-light text-center">
                Or email us at{' '}
                <a href={`mailto:${contactEmail}`} className="text-primary hover:underline">
                  {contactEmail}
                </a>
              </p>
            </form>
          </div>
        )}
      </div>
    </section>
  )
}

