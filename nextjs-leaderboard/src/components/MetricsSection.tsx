interface MetricCardProps {
  title: string;
  description: string;
}

function MetricCard({ title, description }: MetricCardProps) {
  return (
    <div className="rounded-[var(--r-ctrl)] border border-[var(--line)] bg-[var(--surface-2)] p-4">
      <div className="font-semibold text-[var(--ink)] text-[14px] mb-1">
        {title}
      </div>
      <div className="text-[12.5px] text-[var(--ink-2)] leading-relaxed">
        {description}
      </div>
    </div>
  );
}

export default function MetricsSection() {
  const metrics = [
    { title: "F1 Score", description: "Harmonic mean of precision and recall." },
    { title: "Recall", description: "Fraction of true alignments recovered." },
    {
      title: "Precision",
      description: "Fraction of predicted alignments that are correct.",
    },
  ];

  return (
    <div className="card p-5 sm:p-6">
      <h3 className="font-semibold text-[var(--ink)] text-[15px] mb-4">
        Evaluation metrics
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {metrics.map((m) => (
          <MetricCard key={m.title} {...m} />
        ))}
      </div>
    </div>
  );
}
