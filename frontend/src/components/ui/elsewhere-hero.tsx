"use client"

import React, { useEffect, useState, type CSSProperties } from "react"
import { Search, CornerDownLeft, Landmark, ChevronDown } from "lucide-react"

export interface ElsewhereLink {
  label: string
  href: string
}

export interface ElsewhereHeroProps {
  brand?: string
  headline?: string
  subline?: string
  signature?: { name: string; url: string } | false
  className?: string
  style?: CSSProperties
  onSearchSubmit?: (query: string) => void
}

const SANS = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

export default function ElsewhereHero({
  brand = "BharatVault",
  headline = "India's Sovereign Document Intelligence.",
  subline = "159,977 verified excerpts indexed from official central gazettes, statutory acts, and ministerial notifications.",
  className,
  style,
  onSearchSubmit,
}: ElsewhereHeroProps) {
  const [shown, setShown] = useState(false)
  const [heroPrompt, setHeroPrompt] = useState("")

  useEffect(() => {
    const raf = requestAnimationFrame(() => setShown(true))
    return () => cancelAnimationFrame(raf)
  }, [])

  const handleHeroSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!heroPrompt.trim()) return

    if (onSearchSubmit) {
      onSearchSubmit(heroPrompt)
    }

    const consoleEl = document.getElementById("console") || document.getElementById("vault-console")
    if (consoleEl) {
      consoleEl.scrollIntoView({ behavior: "smooth" })
    }
  }

  const rise = (i: number): CSSProperties => ({
    opacity: shown ? 1 : 0,
    transform: shown ? "none" : "translateY(16px)",
    transition: `opacity 0.9s ease ${0.1 + i * 0.12}s, transform 0.9s cubic-bezier(.2,.7,.2,1) ${0.1 + i * 0.12}s`,
  })

  return (
    <section
      id="hero"
      className={`relative min-h-screen w-full flex flex-col justify-between overflow-hidden bg-transparent text-[#f5f3ec] ${className || ""}`}
      aria-label={brand}
      style={{
        fontFamily: SANS,
        ...style,
      }}
    >
      {/* HERO COPY & LOCKED AND LOADED PROMPT BOX */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-center text-center px-4 sm:px-6 max-w-4xl mx-auto pt-28 pb-12 w-full">
        {/* Sovereign Heritage Badge */}
        <div
          className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/10 border border-white/20 backdrop-blur-md mb-6 shadow-sm"
          style={rise(0)}
        >
          <Landmark className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-[11px] sm:text-xs font-mono tracking-widest uppercase text-amber-200">
            Republic of India • Sovereign Intelligence
          </span>
        </div>

        {/* Main Headline */}
        <h1
          className="ew-title max-w-3xl leading-[1.04] text-balance"
          style={rise(1)}
        >
          {headline}
        </h1>

        {/* Subline */}
        <p
          className="ew-sub mt-4 sm:mt-5 max-w-2xl mx-auto"
          style={rise(2)}
        >
          {subline}
        </p>

        {/* LOCKED AND LOADED PROMPT BOX */}
        <div className="w-full max-w-2xl mt-8" style={rise(3)}>
          <div className="bg-[#12160f]/80 backdrop-blur-2xl border border-white/20 rounded-2xl p-2.5 sm:p-3 shadow-2xl transition-all hover:border-white/30 text-left">
            <form onSubmit={handleHeroSubmit} className="relative flex items-center">
              <Search className="w-4 sm:w-5 h-4 sm:h-5 text-amber-400/80 ml-3.5 absolute pointer-events-none" />
              <input
                type="text"
                value={heroPrompt}
                onChange={(e) => setHeroPrompt(e.target.value)}
                placeholder="Ask anything from verified public gazettes and circulars..."
                className="w-full bg-white/5 border border-white/10 rounded-xl pl-11 pr-28 py-3.5 text-xs sm:text-sm text-white placeholder-[#f5f3ec]/40 outline-none focus:border-amber-400/50 transition-all font-sans"
              />
              <button
                type="submit"
                disabled={!heroPrompt.trim()}
                className="absolute right-1.5 sm:right-2 bg-[#f5f3ec] text-[#0a0d08] font-semibold text-xs px-3 sm:px-4 py-2 sm:py-2.5 rounded-lg hover:bg-white disabled:opacity-30 transition-all flex items-center gap-1.5 cursor-pointer shadow-md"
              >
                <span>Query</span>
                <CornerDownLeft className="w-3.5 h-3.5" />
              </button>
            </form>

            {/* Curated Prompt Chips */}
            <div className="flex flex-wrap items-center gap-1.5 sm:gap-2 mt-2.5 pt-2.5 border-t border-white/10 text-xs px-1">
              <span className="text-[#f5f3ec]/50 text-[10px] sm:text-[11px] font-medium mr-1">
                Suggested:
              </span>
              {[
                "HAL apprentice walk-in date",
                "Central Motor Vehicles Rules",
                "Kerala Free WiFi KFi Project",
                "National Pension Scheme rules",
              ].map((item, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setHeroPrompt(item)
                    if (onSearchSubmit) onSearchSubmit(item)
                    const el = document.getElementById("console") || document.getElementById("vault-console")
                    el?.scrollIntoView({ behavior: "smooth" })
                  }}
                  className="bg-white/5 hover:bg-white/10 text-[#f5f3ec]/80 hover:text-white px-2.5 py-1 rounded-full border border-white/10 hover:border-white/20 transition-all cursor-pointer text-[10px] sm:text-[11px]"
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-center gap-3 sm:gap-4 mt-3 text-[11px] text-[#f5f3ec]/55 font-mono">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Pinecone Serverless
            </span>
            <span>•</span>
            <span>Verifiable In-Line Citations</span>
            <span>•</span>
            <span>Zero Speculation</span>
          </div>
        </div>
      </div>

      {/* Hero Bottom Scroll Hint */}
      <div className="relative z-10 w-full px-6 py-4 flex items-center justify-center border-t border-white/5 bg-transparent">
        <a
          href="#overview"
          className="inline-flex items-center gap-2 text-xs font-mono text-[#f5f3ec]/50 hover:text-[#f5f3ec] transition-colors"
        >
          <span>Explore Architecture & Overview</span>
          <ChevronDown className="w-3.5 h-3.5 animate-bounce" />
        </a>
      </div>
    </section>
  )
}
