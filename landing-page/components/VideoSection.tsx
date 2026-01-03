export default function VideoSection() {
  // YouTube video ID from https://youtu.be/rh4UtH3IVH8
  const youtubeVideoId = 'rh4UtH3IVH8'
  const youtubeEmbedUrl = `https://www.youtube.com/embed/${youtubeVideoId}?rel=0`

  return (
    <section id="video-overview" className="py-16 md:py-24 bg-white scroll-mt-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl md:text-4xl font-bold text-navy mb-4">
            2-minute overview
          </h2>
        </div>

        {/* YouTube Video Embed */}
        <div className="rounded-lg overflow-hidden shadow-lg aspect-video">
          <iframe
            src={youtubeEmbedUrl}
            className="w-full h-full"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
            title="EasyWIF Demo Video"
          ></iframe>
        </div>
      </div>
    </section>
  )
}

