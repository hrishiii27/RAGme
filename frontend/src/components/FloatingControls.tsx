import React from "react"

interface FloatingControlsProps {
  isNight: boolean
  onToggleNight: () => void
  muted: boolean
  onToggleMuted: () => void
}

export default function FloatingControls({
  isNight,
  onToggleNight,
  muted,
  onToggleMuted,
}: FloatingControlsProps) {
  return (
    <div
      className="fixed bottom-6 left-6 z-50 flex items-center gap-2.5 p-1.5 rounded-full bg-[#12160f]/75 border border-white/15 backdrop-blur-2xl shadow-2xl transition-all"
      role="region"
      aria-label="Ambient Controls"
    >
      {/* Day / Night Toggle */}
      <button
        onClick={onToggleNight}
        aria-label={isNight ? "Switch to daytime" : "Switch to nighttime"}
        title={isNight ? "Day mode" : "Night mode"}
        className="w-10 h-10 rounded-full flex items-center justify-center bg-white/5 hover:bg-white/15 border border-white/15 text-[#f5f3ec] transition-all cursor-pointer"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          {isNight ? (
            <path
              d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinejoin="round"
            />
          ) : (
            <>
              <circle cx="12" cy="12" r="4.5" stroke="currentColor" strokeWidth="1.6" />
              <path
                d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8 6 18M18 6l1.8-1.8"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </>
          )}
        </svg>
      </button>

      {/* Soundscape Ambient Audio Toggle */}
      <button
        onClick={onToggleMuted}
        aria-label={muted ? "Unmute ambient soundscape" : "Mute ambient soundscape"}
        title={muted ? "Turn on ambient soundscape" : "Mute ambient soundscape"}
        className="w-10 h-10 rounded-full flex items-center justify-center bg-white/5 hover:bg-white/15 border border-white/15 text-[#f5f3ec] transition-all cursor-pointer relative"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M4 9.5v5h3.6L13 19V5L7.6 9.5H4Z"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinejoin="round"
          />
          {muted ? (
            <path
              d="M16 9.5 20 14.5M20 9.5 16 14.5"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          ) : (
            <path
              d="M16.2 8.8a5 5 0 0 1 0 6.4M18.6 6.6a8.5 8.5 0 0 1 0 10.8"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          )}
        </svg>
      </button>

      {/* Mode Status Pill */}
      <span className="hidden sm:inline-block text-[11px] font-mono text-[#f5f3ec]/60 px-2.5 pr-3 select-none">
        {isNight ? "Night Ambience" : "Day Ambience"}
      </span>
    </div>
  )
}
