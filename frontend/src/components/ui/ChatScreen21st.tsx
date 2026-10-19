import React, { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "motion/react"
import {
  Sparkles,
  ArrowRight,
  ThumbsUp,
  ThumbsDown,
  Check,
  RotateCcw,
  Send,
  Database,
  Layers,
  FileText,
  CornerDownLeft,
  ChevronLeft,
  Copy,
  CheckCheck,
  ShieldCheck,
  Cpu,
  Info,
} from "lucide-react"

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: any[]
  standalone_question?: string
  follow_ups?: string[]
  trace_id?: string
  latency?: number
}

interface ChatScreen21stProps {
  messages: ChatMessage[]
  loading: boolean
  query: string
  setQuery: (q: string) => void
  onSend: (prompt: string) => void
  onReset: () => void
  onBackToLanding: () => void
  onOpenSources: (sources: any[], initialIndex?: number) => void
  handleVote: (msgId: string, vote: "positive" | "negative", msg: ChatMessage) => void
  submitFeedback: (vote: string, critique: string, msg: ChatMessage) => void
  feedbackGiven: Record<string, string>
  showCritiqueFor: string | null
  setShowCritiqueFor: (id: string | null) => void
  critiqueText: string
  setCritiqueText: (text: string) => void
  renderFormattedText: (text: string, sources?: any[]) => React.ReactNode
  stats?: any
  isNight?: boolean
}

const SUGGESTED_QUERIES = [
  "HAL apprentice walk-in date & eligibility",
  "Central Motor Vehicles Rules penalty schedule",
  "National Pension Scheme (NPS) revised rules",
  "Kerala Free WiFi KFi Project circular",
]

export default function ChatScreen21st({
  messages,
  loading,
  query,
  setQuery,
  onSend,
  onReset,
  onBackToLanding,
  onOpenSources,
  handleVote,
  submitFeedback,
  feedbackGiven,
  showCritiqueFor,
  setShowCritiqueFor,
  critiqueText,
  setCritiqueText,
  renderFormattedText,
  stats,
}: ChatScreen21stProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const isAutoScrollActive = useRef(true)

  // Track user scroll position: if the user scrolls up to read earlier history, pause auto-scroll
  useEffect(() => {
    const handleScroll = () => {
      const threshold = 180
      const scrollPos = window.scrollY + window.innerHeight
      const bottomPos = document.documentElement.scrollHeight
      isAutoScrollActive.current = bottomPos - scrollPos < threshold
    }
    window.addEventListener("scroll", handleScroll, { passive: true })
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  // Jitter-free scrolling: During streaming, scroll instantly via rAF without fighting animation frames
  useEffect(() => {
    if (!isAutoScrollActive.current) return

    if (loading) {
      // Use instant scrolling with rAF to eliminate all browser scroll-jitter
      const frame = requestAnimationFrame(() => {
        window.scrollTo({
          top: document.documentElement.scrollHeight,
          behavior: "auto",
        })
      })
      return () => cancelAnimationFrame(frame)
    } else {
      // Gentle smooth scroll only when streaming finishes or on initial query submission
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages, loading])

  // Auto-resize textarea
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuery(e.target.value)
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`
    }
  }

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!query.trim() || loading) return
    onSend(query.trim())
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const totalVectors = stats?.total_chunks || 681220

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="relative min-h-screen flex flex-col justify-between pt-24 pb-36 px-4 sm:px-6 max-w-4xl mx-auto w-full z-10"
    >
      {/* ── Chat Top Command Bar ── */}
      <div className="sticky top-20 z-20 mb-6 flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-2xl bg-[#0f140d]/85 backdrop-blur-xl border border-white/10 shadow-[0_4px_24px_rgba(0,0,0,0.4)]">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToLanding}
            className="flex items-center gap-1.5 text-xs font-mono text-[#f5f3ec]/70 hover:text-white px-2.5 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors cursor-pointer"
            title="Return to landing overview"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
            <span>Overview</span>
          </button>

          <div className="h-4 w-px bg-white/10 hidden sm:block" />

          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-semibold text-white tracking-tight">Active RAG Session</span>
            <span className="hidden sm:inline-block text-[11px] font-mono text-amber-300/80">
              • {totalVectors.toLocaleString()} Vectors
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 text-xs font-mono text-[#f5f3ec]/70 hover:text-white px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors cursor-pointer"
            title="Start new topic"
          >
            <RotateCcw className="w-3 h-3" />
            <span>New Query</span>
          </button>
        </div>
      </div>

      {/* ── Message Thread Stream ── */}
      <div className="flex-1 space-y-6">
        {messages.length === 0 && (
          <div className="text-center py-24 max-w-lg mx-auto space-y-4">
            <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-400/30 flex items-center justify-center mx-auto text-amber-300 shadow-[0_0_24px_rgba(245,158,11,0.2)]">
              <Sparkles className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-white tracking-tight">Sovereign Document Intelligence</h3>
            <p className="text-xs text-[#f5f3ec]/65 leading-relaxed font-sans">
              Ask any question across {totalVectors.toLocaleString()} verified statutory acts, central gazettes, and official notifications. Every response is grounded with verbatim citation lineage.
            </p>
          </div>
        )}

        {messages.map((msg) => {
          const isUser = msg.role === "user"

          if (isUser) {
            return (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                className="flex justify-end"
              >
                <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tr-sm bg-gradient-to-br from-amber-500/20 to-amber-700/10 border border-amber-400/30 px-5 py-3.5 shadow-lg backdrop-blur-md">
                  <p className="text-sm font-medium text-white leading-relaxed">{msg.content}</p>
                </div>
              </motion.div>
            )
          }

          // Assistant Response Card
          return (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="rounded-2xl border border-white/15 bg-[#10140e]/85 backdrop-blur-xl p-5 sm:p-7 space-y-4 shadow-2xl relative overflow-hidden"
            >
              {/* Header: Verified badge & Source count */}
              <div className="flex items-center justify-between border-b border-white/10 pb-3.5 gap-2">
                <div className="flex items-center gap-2.5">
                  <div className="w-6 h-6 rounded-full bg-emerald-500/15 border border-emerald-400/30 flex items-center justify-center text-emerald-400">
                    <ShieldCheck className="w-3.5 h-3.5" />
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-amber-300 tracking-wider uppercase block">
                      Sovereign Verification
                    </span>
                    {msg.standalone_question && (
                      <span className="hidden md:inline-block text-[10px] font-mono text-[#f5f3ec]/40 truncate max-w-sm">
                        Intent: "{msg.standalone_question}"
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleCopy(msg.id, msg.content)}
                    className="p-1.5 rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
                    title="Copy response text"
                  >
                    {copiedId === msg.id ? (
                      <CheckCheck className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>

                  {msg.sources && msg.sources.length > 0 && (
                    <button
                      onClick={() => onOpenSources(msg.sources!, 1)}
                      className="text-xs font-mono font-medium text-amber-300 bg-amber-400/10 hover:bg-amber-400/20 border border-amber-400/30 px-3 py-1 rounded-lg transition-colors cursor-pointer flex items-center gap-1.5"
                    >
                      <FileText className="w-3 h-3" />
                      <span>{msg.sources.length} Sources</span>
                      <ArrowRight className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>

              {/* Real-time Streaming State or Formatted Text */}
              <div className="text-sm leading-relaxed text-[#f5f3ec]/90 font-sans">
                {renderFormattedText(msg.content, msg.sources)}
                {loading && msg.id === messages[messages.length - 1].id && (
                  <span className="inline-block w-2 h-4 ml-1 bg-amber-400 animate-pulse align-middle" />
                )}
              </div>

              {/* Grounded Citation Dossier Cards */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="pt-4 border-t border-white/10 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-bold text-amber-300/80 uppercase tracking-widest flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                      <span>Verified Sovereign Sources ({msg.sources.length}):</span>
                    </span>
                    <button
                      onClick={() => onOpenSources(msg.sources!, 1)}
                      className="text-[11px] font-mono text-stone-400 hover:text-amber-300 transition-colors flex items-center gap-1 cursor-pointer"
                    >
                      <span>View All Excerpts</span>
                      <ArrowRight className="w-3 h-3" />
                    </button>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {msg.sources.slice(0, 4).map((s, idx) => {
                      let hostname = "gov.in"
                      try {
                        if (s.url) hostname = new URL(s.url).hostname
                      } catch (_) {}

                      return (
                        <button
                          key={idx}
                          type="button"
                          onClick={() => onOpenSources(msg.sources!, idx + 1)}
                          className="group p-3 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 hover:border-amber-400/40 transition-all text-left cursor-pointer flex flex-col justify-between gap-1.5 shadow-sm"
                        >
                          <div className="flex items-center justify-between gap-2 w-full">
                            <span className="inline-flex items-center gap-1.5 text-[11px] font-bold text-amber-300">
                              <span className="w-4 h-4 rounded-full bg-amber-400/20 text-amber-300 text-[10px] flex items-center justify-center font-bold">
                                {idx + 1}
                              </span>
                              <span>[{idx + 1}]</span>
                            </span>

                            {s.category && (
                              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                                {s.category}
                              </span>
                            )}
                          </div>

                          <div className="text-xs font-semibold text-white group-hover:text-amber-200 transition-colors line-clamp-1">
                            {s.title || "Official Public Document"}
                          </div>

                          <div className="flex items-center justify-between text-[10px] font-mono text-stone-400 w-full pt-1 border-t border-white/5">
                            <span className="truncate max-w-[140px] text-[#f5f3ec]/60">
                              {s.department || hostname}
                            </span>
                            <span className="text-amber-400/80 group-hover:translate-x-0.5 transition-transform">
                              Inspect ↗
                            </span>
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Follow-up Questions Chips */}
              {msg.follow_ups && msg.follow_ups.length > 0 && (
                <div className="pt-3.5 border-t border-white/10 space-y-2">
                  <span className="text-[10px] font-mono font-bold text-amber-300/80 uppercase tracking-widest flex items-center gap-1.5">
                    <Sparkles className="w-3 h-3 text-amber-400" />
                    <span>Suggested Follow-Ups:</span>
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {msg.follow_ups.map((q, idx) => (
                      <button
                        key={idx}
                        onClick={() => onSend(q)}
                        disabled={loading}
                        className="text-xs font-sans bg-white/5 hover:bg-white/10 border border-white/10 text-[#f5f3ec]/80 hover:text-white px-3 py-1.5 rounded-xl transition-all cursor-pointer text-left hover:border-amber-400/30 flex items-center gap-1.5"
                      >
                        <span className="text-amber-400">↳</span>
                        <span>{q}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Citizen Feedback Footer */}
              {!loading && msg.content && (
                <div className="pt-3.5 border-t border-white/10 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-[#f5f3ec]/60">
                  {feedbackGiven[msg.id] ? (
                    <div className="flex items-center gap-1.5 text-emerald-400 font-mono text-[11px]">
                      <Check className="w-3.5 h-3.5" />
                      <span>Audited in Langfuse telemetry</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-[11px] text-[#f5f3ec]/60">Was this accurate?</span>
                      <button
                        onClick={() => handleVote(msg.id, "positive", msg)}
                        className="p-1 rounded hover:bg-white/10 text-white/70 hover:text-emerald-400 transition-colors cursor-pointer"
                        title="Grounded and accurate"
                      >
                        <ThumbsUp className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleVote(msg.id, "negative", msg)}
                        className="p-1 rounded hover:bg-white/10 text-white/70 hover:text-red-400 transition-colors cursor-pointer"
                        title="Missing or inaccurate"
                      >
                        <ThumbsDown className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}

                  <div className="flex items-center gap-2 font-mono text-[10px] text-[#f5f3ec]/40">
                    <Database className="w-3 h-3 text-amber-400/70" />
                    <span>Pinecone Serverless Hybrid RRF</span>
                  </div>
                </div>
              )}

              {/* Critique Feedback Drawer */}
              {showCritiqueFor === msg.id && (
                <div className="p-3.5 rounded-xl bg-white/5 border border-white/15 space-y-2 mt-2">
                  <textarea
                    value={critiqueText}
                    onChange={(e) => setCritiqueText(e.target.value)}
                    placeholder="What specific clause or detail was missing or inaccurate?..."
                    className="w-full text-xs p-3 rounded-lg bg-black/60 border border-white/10 text-white outline-none focus:border-amber-400/50 resize-none h-20 font-sans"
                  />
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => setShowCritiqueFor(null)}
                      className="text-xs px-3 py-1 text-white/60 hover:text-white cursor-pointer"
                    >
                      Skip
                    </button>
                    <button
                      onClick={() => submitFeedback("negative", critiqueText, msg)}
                      className="text-xs px-3.5 py-1.5 bg-amber-400 text-black font-semibold rounded-lg hover:bg-amber-300 transition-colors cursor-pointer"
                    >
                      Submit Audit
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          )
        })}

        {/* Live Loading Thinking Skeleton Indicator */}
        {loading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-amber-400/20 bg-[#10140e]/90 backdrop-blur-xl p-5 sm:p-6 space-y-3"
          >
            <div className="flex items-center gap-2 text-xs font-mono text-amber-300">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
              <span>Grounded Retrieval in Progress...</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-2">
              <div className="flex items-center gap-2 p-2 rounded-lg bg-white/5 border border-white/10 text-[11px] font-mono text-[#f5f3ec]/70">
                <Database className="w-3.5 h-3.5 text-amber-400" />
                <span>681k Pinecone Vectors</span>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-lg bg-white/5 border border-white/10 text-[11px] font-mono text-[#f5f3ec]/70">
                <Layers className="w-3.5 h-3.5 text-emerald-400" />
                <span>BM25 Keyword Fusion</span>
              </div>
              <div className="flex items-center gap-2 p-2 rounded-lg bg-white/5 border border-white/10 text-[11px] font-mono text-[#f5f3ec]/70">
                <Cpu className="w-3.5 h-3.5 text-blue-400" />
                <span>Few-Shot Prompt Calibrated</span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ── Fixed Floating 21st.dev Chat Dock at the Bottom ── */}
      <div className="fixed bottom-6 left-0 right-0 z-40 px-4 sm:px-6 pointer-events-none">
        <div className="max-w-3xl mx-auto pointer-events-auto">
          {/* Quick Query Pill Chips */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-2 scrollbar-none mb-1 px-1">
            {SUGGESTED_QUERIES.map((item, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => onSend(item)}
                disabled={loading}
                className="whitespace-nowrap bg-[#0f140d]/90 hover:bg-white/10 text-[#f5f3ec]/75 hover:text-white border border-white/15 px-3 py-1 rounded-full text-[11px] font-mono backdrop-blur-md transition-all cursor-pointer shadow-md hover:border-amber-400/40"
              >
                {item}
              </button>
            ))}
          </div>

          {/* Floating Input Dock */}
          <div className="bg-[#121710]/95 backdrop-blur-2xl border border-white/20 hover:border-white/30 rounded-2xl p-2 sm:p-2.5 shadow-[0_8px_32px_rgba(0,0,0,0.6)] transition-all">
            <form onSubmit={handleSubmit} className="relative flex items-center">
              <textarea
                ref={textareaRef}
                value={query}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                rows={1}
                placeholder="Ask about official acts, circulars, notifications, apprentice lists..."
                className="w-full bg-white/5 border border-white/10 rounded-xl pl-4 pr-24 py-3 text-xs sm:text-sm text-white placeholder-[#f5f3ec]/40 outline-none focus:border-amber-400/50 transition-all font-sans resize-none leading-relaxed"
                style={{ maxHeight: "140px" }}
              />

              <div className="absolute right-2 flex items-center gap-1.5">
                <button
                  type="submit"
                  disabled={!query.trim() || loading}
                  className="bg-amber-400 hover:bg-amber-300 text-[#0a0d08] font-bold p-2.5 rounded-xl disabled:opacity-30 disabled:pointer-events-none transition-all flex items-center justify-center cursor-pointer shadow-md"
                  title="Send inquiry"
                >
                  <Send className="w-4 h-4 text-[#0a0d08]" />
                </button>
              </div>
            </form>

            <div className="flex items-center justify-between px-2 pt-2 border-t border-white/5 text-[10px] font-mono text-[#f5f3ec]/40">
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                <span>Grounded in Pinecone Cloud</span>
              </span>
              <span>Press Enter to send, Shift+Enter for newline</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
