import Image from 'next/image'
import { contactEmail, linkedInUrl } from '@/app/constants'

export default function Footer() {
  return (
    <footer className="bg-navy text-white py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          {/* Logo and Summary */}
          <div>
            <Image
              src="/ezw logo_black.png"
              alt="EasyWIF Logo"
              width={100}
              height={33}
              className="h-8 w-auto mb-4 brightness-0 invert"
            />
            <p className="text-gray-300 text-sm">
              Scenario planning in minutes — not weeks. EasyWIF helps teams model planning changes quickly.
            </p>
          </div>

          {/* Links */}
          <div>
            <h3 className="font-semibold mb-4">Links</h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li>
                <a href="#how-it-works" className="hover:text-white transition-colors">
                  How it Works
                </a>
              </li>
              <li>
                <a href="#product" className="hover:text-white transition-colors">
                  Product
                </a>
              </li>
              <li>
                <a href="#use-cases" className="hover:text-white transition-colors">
                  Use Cases
                </a>
              </li>
              <li>
                <a href="#faq" className="hover:text-white transition-colors">
                  FAQ
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="font-semibold mb-4">Contact</h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li>
                <a href={`mailto:${contactEmail}`} className="hover:text-white transition-colors">
                  {contactEmail}
                </a>
              </li>
              <li>
                <a href={linkedInUrl} target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">
                  LinkedIn
                </a>
              </li>
              <li>
                <a href="#book-demo" className="hover:text-white transition-colors">
                  Book Demo
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-700 pt-8 flex flex-col md:flex-row justify-between items-center text-sm text-gray-400">
          <p>&copy; {new Date().getFullYear()} EasyWIF. All rights reserved.</p>
          <div className="flex gap-6 mt-4 md:mt-0">
            <a href="#" className="hover:text-white transition-colors">
              Privacy Policy
            </a>
            <a href={`mailto:${contactEmail}`} className="hover:text-white transition-colors">
              Contact
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}

