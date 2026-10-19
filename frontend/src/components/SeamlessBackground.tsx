import React from "react"

interface SeamlessBackgroundProps {
  activeSection: string
  isNight: boolean
}

interface BackgroundLayer {
  id: string
  daySrc: string
  nightSrc?: string
}

const LAYERS: BackgroundLayer[] = [
  {
    id: "hero",
    daySrc: "/parliament-day.jpg",
    nightSrc: "/parliament-night.jpg",
  },
  {
    id: "overview",
    daySrc: "/bg-overview.jpg",
    nightSrc: "/bg-overview.jpg",
  },
  {
    id: "console",
    daySrc: "/parliament-night.jpg",
    nightSrc: "/parliament-night.jpg",
  },
  {
    id: "architecture",
    daySrc: "/bg-architecture.jpg",
    nightSrc: "/bg-architecture.jpg",
  },
  {
    id: "methodology",
    daySrc: "/bg-architecture.jpg",
    nightSrc: "/bg-architecture.jpg",
  },
  {
    id: "benchmarks",
    daySrc: "/bg-benchmarks.jpg",
    nightSrc: "/bg-benchmarks.jpg",
  },
  {
    id: "corpus",
    daySrc: "/bg-corpus.jpg",
    nightSrc: "/bg-corpus.jpg",
  },
]

export default function SeamlessBackground({
  activeSection,
  isNight,
}: SeamlessBackgroundProps) {
  return (
    <div
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden bg-[#0a0d08]"
      aria-hidden="true"
    >
      {LAYERS.map((layer) => {
        const isActive = activeSection === layer.id
        const imageSrc =
          isNight && layer.nightSrc ? layer.nightSrc : layer.daySrc

        return (
          <div
            key={layer.id}
            className="absolute inset-0 bg-cover bg-center transition-opacity duration-1000 ease-in-out"
            style={{
              backgroundImage: `url(${imageSrc})`,
              opacity: isActive ? 1 : 0,
              animation: "bv-subtle-zoom 32s infinite alternate ease-in-out",
            }}
          />
        )
      })}

      {/* Master Cinematic Vignette Overlay */}
      <div
        className="absolute inset-0 transition-opacity duration-700"
        style={{
          background: isNight
            ? "linear-gradient(180deg, rgba(10,13,8,0.85) 0%, rgba(10,13,8,0.60) 40%, rgba(10,13,8,0.92) 100%)"
            : "linear-gradient(180deg, rgba(10,13,8,0.76) 0%, rgba(10,13,8,0.48) 40%, rgba(10,13,8,0.86) 100%)",
        }}
      />
    </div>
  )
}
