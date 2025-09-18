interface DatasetCardProps {
  title: string;
  description: string;
  iconColor: string;
  hoverClass: string;
  bgColor: string;
  borderColor: string;
  type: string;
  challenge: string;
  goal: string;
  details: string;
}

function DatasetCard({
  title,
  description,
  iconColor,
  hoverClass,
  bgColor,
  borderColor,
  type,
  challenge,
  goal,
  details,
}: DatasetCardProps) {
  return (
    <div
      className={`bg-[var(--background)] border border-[var(--border)] rounded-xl p-6 hover:shadow-lg ${hoverClass} transition-all`}
    >
      <div className="flex items-center space-x-3 mb-4">
        <div
          className={`w-12 h-12 ${bgColor} rounded-xl flex items-center justify-center`}
        >
          <div className={`w-6 h-6 ${iconColor} rounded-lg`}></div>
        </div>
        <h3 className="font-bold text-[var(--foreground)] text-lg">{title}</h3>
      </div>
      <p className="text-sm text-[var(--toss-light-gray)] leading-relaxed mb-4">
        {description}
      </p>
      <div className="space-y-2 text-xs text-[var(--toss-light-gray)]">
        <div className="flex justify-between">
          <span>Type:</span>
          <span className="font-medium text-[var(--foreground)]">{type}</span>
        </div>
        <div className="flex justify-between">
          <span>Challenge:</span>
          <span className="font-medium text-[var(--foreground)]">
            {challenge}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Goal:</span>
          <span className="font-medium text-[var(--foreground)]">{goal}</span>
        </div>
      </div>
      <div className={`mt-4 p-3 ${bgColor} rounded-lg border ${borderColor}`}>
        <p className="text-xs text-[var(--toss-light-gray)]">{details}</p>
      </div>
    </div>
  );
}

export default function DatasetCards() {
  const datasets = [
    {
      title: "MALIDUP",
      description: "True homologs with low sequence identity",
      iconColor: "bg-[var(--toss-blue)]",
      hoverClass: "hover:border-blue-500/20",
      bgColor: "bg-blue-50 dark:bg-blue-900/10",
      borderColor: "border-blue-100 dark:border-blue-900/30",
      type: "True Homologs",
      challenge: "Low Sequence Identity",
      goal: "High Recall",
      details:
        "Contains protein pairs that are evolutionarily related but have low sequence similarity, often from domain duplication events.",
    },
    {
      title: "MALISAM",
      description: "Non-homologous structural analogs",
      iconColor: "bg-red-500",
      hoverClass: "hover:border-red-500/20",
      bgColor: "bg-red-50 dark:bg-red-900/10",
      borderColor: "border-red-100 dark:border-red-900/30",
      type: "Structural Analogs",
      challenge: "Convergent Evolution",
      goal: "Low False Positives",
      details:
        "Contains protein pairs with similar structures but no evolutionary relationship, testing specificity.",
    },
    {
      title: "SABmark",
      description: "Remote homologs from SCOP superfamilies",
      iconColor: "bg-green-500",
      hoverClass: "hover:border-green-500/20",
      bgColor: "bg-green-50 dark:bg-green-900/10",
      borderColor: "border-green-100 dark:border-green-900/30",
      type: "Remote Homologs",
      challenge: "Distant Relationships",
      goal: "Balanced Performance",
      details:
        "Challenging cases of remote homologs grouped by SCOP superfamilies with structural alignments as ground truth.",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {datasets.map((dataset) => (
        <DatasetCard key={dataset.title} {...dataset} />
      ))}
    </div>
  );
}
