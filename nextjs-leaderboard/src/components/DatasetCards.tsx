interface DatasetCardProps {
  title: string;
  accent: string;
  description: string;
  type: string;
  challenge: string;
  goal: string;
  details: string;
}

function DatasetCard({
  title,
  accent,
  description,
  type,
  challenge,
  goal,
  details,
}: DatasetCardProps) {
  return (
    <div className="card p-5 transition-colors hover:border-[var(--line-2)]">
      <div className="flex items-center gap-2.5 mb-3">
        <span
          className="w-2.5 h-2.5 rounded-full shrink-0"
          style={{ backgroundColor: accent }}
        />
        <h3 className="font-semibold text-[var(--ink)] text-[15px]">{title}</h3>
      </div>
      <p className="text-[13px] text-[var(--ink-2)] leading-relaxed mb-4">
        {description}
      </p>
      <dl className="space-y-1.5 text-[13px]">
        {[
          ["Type", type],
          ["Challenge", challenge],
          ["Goal", goal],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3">
            <dt className="text-[var(--ink-3)]">{k}</dt>
            <dd className="font-medium text-[var(--ink)] text-right">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 pt-4 border-t border-[var(--line)] text-[12.5px] text-[var(--ink-3)] leading-relaxed">
        {details}
      </p>
    </div>
  );
}

export default function DatasetCards() {
  const datasets: DatasetCardProps[] = [
    {
      title: "MALIDUP",
      accent: "var(--series-ot)",
      description: "True homologs with low sequence identity.",
      type: "True homologs",
      challenge: "Low sequence identity",
      goal: "High recall",
      details:
        "Protein pairs that are evolutionarily related but share low sequence similarity, often from domain duplication events.",
    },
    {
      title: "MALISAM",
      accent: "var(--series-trad)",
      description: "Non-homologous structural analogs.",
      type: "Structural analogs",
      challenge: "Convergent evolution",
      goal: "Low false positives",
      details:
        "Protein pairs with similar structures but no evolutionary relationship — a test of specificity.",
    },
    {
      title: "SABmark",
      accent: "var(--series-plm)",
      description: "Remote homologs from SCOP superfamilies.",
      type: "Remote homologs",
      challenge: "Distant relationships",
      goal: "Balanced performance",
      details:
        "Challenging remote homologs grouped by SCOP superfamily, with structural alignments as ground truth.",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {datasets.map((d) => (
        <DatasetCard key={d.title} {...d} />
      ))}
    </div>
  );
}
