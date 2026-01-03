'use client'

import { useState } from 'react'
import Image from 'next/image'

export default function VideoSection() {
  const [isPlaying, setIsPlaying] = useState(false)

  const handlePlay = () => {
    setIsPlaying(true)
  }

  // YouTube video ID from https://youtu.be/rh4UtH3IVH8
  const youtubeVideoId = 'rh4UtH3IVH8'
  const youtubeEmbedUrl = `https://www.youtube.com/embed/${youtubeVideoId}?autoplay=1&rel=0`

  return (
    <section id="video-overview" className="py-16 md:py-24 bg-white scroll-mt-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl md:text-4xl font-bold text-navy mb-4">
            2-minute overview
          </h2>
        </div>

        {/* Video Player with Logo Cover */}
        <div className="relative rounded-lg overflow-hidden shadow-lg aspect-video bg-white">
          {!isPlaying ? (
            <div 
              className="absolute inset-0 flex items-center justify-center cursor-pointer bg-white"
              onClick={handlePlay}
            >
              <div className="text-center">
                <Image
                  src="/ezw logo_black.png"
                  alt="EasyWIF Logo"
                  width={200}
                  height={67}
                  className="mx-auto mb-6"
                />
                <div className="flex items-center justify-center">
                  <div className="bg-primary rounded-full p-4 hover:bg-primary-dark transition-colors">
                    <svg 
                      className="w-12 h-12 text-white ml-1" 
                      fill="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  </div>
                </div>
                <p className="text-navy-light mt-4 text-sm">Click to play</p>
              </div>
            </div>
          ) : (
            <iframe
              src={youtubeEmbedUrl}
              className="w-full h-full"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              title="EasyWIF Demo Video"
            ></iframe>
          )}
        </div>
      </div>
    </section>
  )
}

