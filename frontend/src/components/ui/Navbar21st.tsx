import React, { useState } from "react"
import { motion, AnimatePresence } from "motion/react"
import { Landmark, Sparkles, BarChart3, Database, Layers, Menu, X, ArrowUpRight } from "lucide-react"

interface NavItem {
  id: string
  label: string
}

interface Navbar21stProps {
  activeSection: string
  onNavigate: (sectionId: string) => void
  onOpenBenchmarks: () => void
  totalVectors?: number
  isNight?: boolean
}

const NAV_ITEMS: NavItem[] = [
  { id: "hero", label: "Overview" },
  { id: "console", label: "Interactive RAG" },
  { id: "architecture", label: "Architecture" },
  { id: "methodology", label: "Deep-Dives" },
  { id: "benchmarks", label: "Benchmarks" },
  { id: "corpus", label: "Corpus" },
]

export default function Navbar21st({
  activeSection,
  onNavigate,
  onOpenBenchmarks,
  totalVectors = 248041,
}: Navbar21stProps) {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <header className="fixed top-4 left-0 right-0 z-50 px-4 sm:px-6 pointer-events-none">
      <div className="max-w-6xl mx-auto flex items-center justify-between gap-4">
        {/* Brand Pill */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="pointer-events-auto flex items-center gap-3 px-4 py-2 rounded-full border border-white/10 bg-[#0e120c]/85 backdrop-blur-xl shadow-[0_4px_24px_rgba(0,0,0,0.5)] cursor-pointer"
          onClick={() => onNavigate("hero")}
        >
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-amber-500 to-amber-700 flex items-center justify-center text-[#0a0d08] font-bold shadow-[0_0_12px_rgba(245,158,11,0.4)]">
            <Landmark className="w-4 h-4 text-[#0a0d08]" />
          </div>
          <div className="flex flex-col">
            <span className="text-xs font-bold tracking-widest uppercase text-[#f5f3ec]">BharatVault</span>
            <span className="text-[9px] font-mono text-amber-400/90 tracking-tight flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping inline-block" />
              {(totalVectors || 248041).toLocaleString()} Vectors Live
            </span>
          </div>
        </motion.div>

        {/* Desktop Nav Items with 21st.dev LayoutId Sliding Pill */}
        <motion.nav
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="hidden md:flex pointer-events-auto items-center p-1.5 rounded-full border border-white/10 bg-[#0e120c]/85 backdrop-blur-xl shadow-[0_4px_24px_rgba(0,0,0,0.5)]"
        >
          {NAV_ITEMS.map((item) => {
            const isActive = activeSection === item.id

            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`relative px-3.5 py-1.5 text-xs font-medium rounded-full transition-colors duration-200 ${
                  isActive ? "text-[#f5f3ec]" : "text-stone-400 hover:text-stone-200"
                }`}
              >
                {isActive && (
                  <motion.div
                    layoutId="activeNavPill"
                    className="absolute inset-0 rounded-full bg-white/10 border border-white/15 shadow-[0_0_12px_rgba(255,255,255,0.06)]"
                    transition={{ type: "spring", stiffness: 380, damping: 30 }}
                  />
                )}
                <span className="relative z-10">{item.label}</span>
              </button>
            )
          })}
        </motion.nav>

        {/* Right Action Button (Benchmark Modal Trigger) */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="pointer-events-auto flex items-center gap-2"
        >
          <button
            onClick={onOpenBenchmarks}
            className="flex items-center gap-2 px-3.5 py-2 rounded-full border border-amber-500/30 bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 text-xs font-semibold backdrop-blur-xl transition-all duration-200 hover:shadow-[0_0_20px_rgba(245,158,11,0.25)]"
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Scorecards</span>
            <span className="px-1.5 py-0.2 text-[9px] font-mono bg-amber-400/20 rounded-full">v2.4</span>
          </button>

          {/* Mobile Menu Toggle */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 rounded-full border border-white/10 bg-[#0e120c]/85 text-stone-300 hover:text-white"
          >
            {mobileOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
        </motion.div>
      </div>

      {/* Mobile Drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="pointer-events-auto md:hidden mt-2 p-3 rounded-2xl border border-white/10 bg-[#0e120c]/95 backdrop-blur-2xl shadow-2xl flex flex-col gap-1 max-w-sm mx-auto"
          >
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  onNavigate(item.id)
                  setMobileOpen(false)
                }}
                className={`px-4 py-2.5 rounded-xl text-left text-xs font-medium transition-colors ${
                  activeSection === item.id
                    ? "bg-amber-500/15 text-amber-300 font-semibold"
                    : "text-stone-300 hover:bg-white/5"
                }`}
              >
                {item.label}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
