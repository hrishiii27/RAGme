import ElsewhereHero from "@/components/ui/elsewhere-hero"

const settings = {
  headline: "BharatVault Sovereign Intelligence",
  subline: "159,977 verified chunks indexed from official government gazettes, circulars, and statutes.",
}

export default function Demo(props: Partial<typeof settings>) {
  const s = { ...settings, ...props }

  return (
    <ElsewhereHero
      headline={s.headline}
      subline={s.subline}
      signature={{
        name: "BharatVault",
        url: "https://github.com/hrishikeshvirupakshi/RAGme",
      }}
    />
  )
}
