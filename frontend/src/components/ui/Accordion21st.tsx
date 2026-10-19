import React, { useState } from "react"
import { motion, AnimatePresence } from "motion/react"
import { ChevronDown, Sparkles, Layers, Cpu, Database, Split } from "lucide-react"

export interface AccordionItemData {
  id: string
  title: string
  badge: string
  badgeColor?: string
  icon: React.ReactNode
  summary: string
  details: React.ReactNode
}

interface AccordionProps {
  items: AccordionItemData[]
  defaultOpenId?: string
}

export default function Accordion21st({ items, defaultOpenId }: AccordionProps) {
  const [openId, setOpenId] = useState<string | null>(defaultOpenId || null)

  const toggle = (id: string) => {
    setOpenId((prev) => (prev === id ? null : id))
  }

  return (
    <div className="space-y-3.5 w-full">
      {items.map((item) => {
        const isOpen = openId === item.id

        return (
          <div
            key={item.id}
            className={`rounded-xl border transition-all duration-300 overflow-hidden ${
              isOpen
                ? "bg-[#141812]/90 border-amber-500/30 shadow-[0_8px_32px_rgba(0,0,0,0.4),0_0_20px_rgba(245,158,11,0.06)]"
                : "bg-[#10140e]/60 border-white/5 hover:border-white/15 hover:bg-[#121610]/80"
            }`}
          >
            <button
              onClick={() => toggle(item.id)}
              className="w-full px-5 py-4 flex items-center justify-between text-left gap-4 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/40"
              aria-expanded={isOpen}
            >
              <div className="flex items-center gap-3.5 flex-1 min-w-0">
                <div
                  className={`p-2 rounded-lg transition-colors shrink-0 ${
                    isOpen ? "bg-amber-500/15 text-amber-400" : "bg-white/5 text-stone-400"
                  }`}
                >
                  {item.icon}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="text-sm font-semibold text-[#f5f3ec] tracking-tight">{item.title}</h3>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                        item.badgeColor || "bg-amber-500/10 text-amber-300 border-amber-500/20"
                      }`}
                    >
                      {item.badge}
                    </span>
                  </div>
                  <p className="text-xs text-stone-400 mt-0.5 line-clamp-1">{item.summary}</p>
                </div>
              </div>

              <motion.div
                animate={{ rotate: isOpen ? 180 : 0 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                className="shrink-0 text-stone-400"
              >
                <ChevronDown className="w-4 h-4" />
              </motion.div>
            </button>

            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div
                  key="content"
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                  className="overflow-hidden"
                >
                  <div className="px-5 pb-5 pt-1 border-t border-white/5 text-xs text-stone-300 leading-relaxed">
                    {item.details}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )
      })}
    </div>
  )
}
