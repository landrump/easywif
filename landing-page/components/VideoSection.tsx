export default function VideoSection() {
  return (
    <section id="video-overview" className="py-16 md:py-24 bg-white scroll-mt-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl md:text-4xl font-bold text-navy mb-4">
            2-minute overview
          </h2>
        </div>

        {/* Video Placeholder */}
        <div className="bg-gray-100 rounded-lg overflow-hidden shadow-lg aspect-video flex items-center justify-center">
          <div className="text-center p-8">
            <p className="text-navy-light mb-4">
              Video embed goes here (Loom / YouTube)
            </p>
            <button className="bg-primary text-white px-6 py-3 rounded-lg hover:bg-primary-dark transition-colors font-medium">
              Play (coming soon)
            </button>
          </div>
        </div>

        {/* Comment for easy iframe insertion */}
        {/* 
          To add video, replace the placeholder div above with:
          <div className="aspect-video rounded-lg overflow-hidden shadow-lg">
            <iframe
              src="YOUR_LOOM_OR_YOUTUBE_EMBED_URL"
              className="w-full h-full"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
          </div>
        */}
      </div>
    </section>
  )
}

