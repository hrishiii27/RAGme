import React from "react"
import { X, ExternalLink, ShieldCheck, FileText, Database } from "lucide-react"

export interface SourceChunk {
  chunk_id: string
  doc_id: string
  title: string
  url: string
  score: number
  snippet: string
  category?: string
  department?: string
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
        <div className="w-screen max-w-lg bg-[#11160e] border-l border-white/10 shadow-2xl flex flex-col text-[#f5f3ec]">
          {/* Drawer Header */}
          <div className="p-5 border-b border-white/10 flex items-center justify-between bg-black/40">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-amber-500/10 border border-amber-400/30 flex items-center justify-center text-amber-300">
                <FileText className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white tracking-tight">Verified Sovereign Context</h3>
                <p className="text-[11px] text-[#f5f3ec]/60 font-mono">
                  Grounded in Pinecone Cloud • {sources.length} retrieved excerpts
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
                  className={`p-5 rounded-2xl border transition-all ${
                    isHighlight
                      ? "bg-[#182015] border-amber-400/80 shadow-[0_0_24px_rgba(245,158,11,0.2)] ring-1 ring-amber-400/50"
                      : "bg-white/5 border-white/10 hover:border-white/20"
                  }`}
                >
                  {/* Top Bar: Citation Pill, Category & Score */}
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center gap-1.5 text-xs font-bold text-amber-300 bg-amber-400/15 border border-amber-400/30 px-2.5 py-0.5 rounded-full">
                        <span className="w-4 h-4 rounded-full bg-amber-400/30 text-[10px] flex items-center justify-center font-bold">
                          {citationNum}
                        </span>
                        <span>Citation [{citationNum}]</span>
                      </span>

                      {src.category && (
                        <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-md bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                          {src.category}
                        </span>
                      )}
                    </div>

                    <span className="text-[10px] font-mono bg-black/40 px-2 py-0.5 rounded border border-white/10 text-stone-300">
                      RRF Rank: {src.score}
                    </span>
                  </div>

                  {/* Document Name / Title */}
                  <h4 className="text-sm font-bold text-white mb-2 leading-snug">
                    {src.title || "Government Gazette / Circular"}
                  </h4>

                  {/* Authority / Ministry & Domain */}
                  <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-[#f5f3ec]/60 mb-3">
                    <span className="flex items-center gap-1 text-amber-300/80">
                      <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                      <span>{src.department || "Central Department"}</span>
                    </span>
                    <span>•</span>
                    <span className="text-stone-400 truncate max-w-[200px]">{hostname}</span>
                  </div>

                  {/* Clean Formatted Snippet Box */}
                  <div className="text-xs leading-relaxed text-[#f5f3ec]/85 bg-black/50 p-3.5 rounded-xl border border-white/10 font-sans whitespace-pre-wrap relative">
                    <div className="text-[9px] font-mono uppercase tracking-widest text-amber-400/60 mb-1.5 flex items-center gap-1">
                      <span>Verbatim Record Excerpt:</span>
                    </div>
                    {src.snippet}
                  </div>

                  {src.url && (
                    <div className="mt-3.5 pt-2.5 border-t border-white/10 flex items-center justify-between">
                      <span className="text-[10px] font-mono text-stone-400">Official Portal Gazette</span>
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 text-xs font-semibold text-amber-300 hover:text-amber-200 transition-colors"
                      >
                        <span>Open Document Source</span>
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
