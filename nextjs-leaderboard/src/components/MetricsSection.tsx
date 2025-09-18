interface MetricCardProps {
  title: string;
  description: string;
}

function MetricCard({ title, description }: MetricCardProps) {
  return (
    <div className="text-center p-4 bg-[var(--background)] border border-[var(--border-light)] rounded-lg hover:border-[var(--border)] transition-colors">
      <div className="font-semibold text-[var(--foreground)] mb-1">{title}</div>
      <div className="text-xs text-[var(--toss-light-gray)]">{description}</div>
    </div>
  );
}

export default function MetricsSection() {
  const metrics = [
    {
      title: "F1 Score",
      description: "Harmonic mean of precision and recall",
    },
    {
      title: "Recall",
      description: "Fraction of true alignments recovered",
    },
    {
      title: "Precision",
      description: "Fraction of predictions that are correct",
    },
  ];

  return (
    <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-6">
      <h3 className="font-bold text-[var(--foreground)] text-lg mb-4 flex items-center">
        <span className="mr-2">📊</span>
        Evaluation Metrics
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {metrics.map((metric) => (
          <MetricCard key={metric.title} {...metric} />
        ))}
      </div>
    </div>
  );
}

