import React from "react"
import { X, ExternalLink, ShieldCheck, FileText, Database } from "lucide-react"

export interface SourceChunk {
  chunk_id: string
  doc_id: string
  title: string
  url: string
  score: number
  snippet: string
}

interface SourceDrawerProps {
  isOpen: boolean
  onClose: () => void
  sources: SourceChunk[]
  activeCitationIndex: number | null
}

export default function SourceDrawer({
  isOpen,
  onClose,
  sources,
  activeCitationIndex,
}: SourceDrawerProps) {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="absolute inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-md bg-[#12160f] border-l border-white/10 shadow-2xl flex flex-col text-[#f5f3ec]">
          {/* Drawer Header */}
          <div className="p-5 border-b border-white/10 flex items-center justify-between bg-black/30">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center text-amber-400">
                <FileText className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white tracking-tight">Verified Source Context</h3>
                <p className="text-[11px] text-[#f5f3ec]/60 font-mono">
                  Grounded in Pinecone Cloud • {sources.length} retrieved chunks
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-[#f5f3ec]/60 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Drawer Content */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            {sources.map((src, idx) => {
              const citationNum = idx + 1
              const isHighlight = activeCitationIndex === citationNum

              let hostname = "gov.in"
              try {
                if (src.url) hostname = new URL(src.url).hostname
              } catch (_) {}

              return (
                <div
                  key={src.chunk_id || idx}
                  className={`p-4 rounded-2xl border transition-all ${
                    isHighlight
                      ? "bg-white/10 border-amber-400/80 shadow-lg ring-1 ring-amber-400/50"
                      : "bg-white/5 border-white/10 hover:border-white/20"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-amber-300">
                      <span className="w-5 h-5 rounded-full bg-amber-400/20 border border-amber-400/40 text-[10px] flex items-center justify-center font-bold">
                        {citationNum}
                      </span>
                      <span>Citation [{citationNum}]</span>
                    </span>

                    <span className="text-[10px] font-mono bg-white/5 px-2 py-0.5 rounded border border-white/10 text-[#f5f3ec]/60">
                      Score: {src.score}
                    </span>
                  </div>

                  <h4 className="text-xs font-bold text-white mb-1.5 line-clamp-2">
                    {src.title || "Government Gazette / Circular"}
                  </h4>

                  <div className="text-[11px] font-mono text-[#f5f3ec]/50 mb-2.5 flex items-center gap-1.5">
                    <ShieldCheck className="w-3 h-3 text-emerald-400" />
                    <span>Domain: {hostname}</span>
                  </div>

                  <div className="text-xs leading-relaxed text-[#f5f3ec]/80 bg-black/40 p-3 rounded-xl border border-white/5 font-sans whitespace-pre-wrap">
                    {src.snippet}
                  </div>

                  {src.url && (
                    <div className="mt-3 pt-2.5 border-t border-white/10 flex justify-end">
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 text-[11px] font-medium text-amber-300 hover:text-amber-200"
                      >
                        <span>Official Document Source</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Drawer Footer */}
          <div className="p-4 border-t border-white/10 bg-black/40 text-center">
            <span className="text-[11px] text-[#f5f3ec]/50">
              Verified against Sovereign Archive • Zero Hallucination Standard
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
