interface MethodCategoryProps {
  color: string;
  title: string;
  description: string;
}

function MethodCategory({ color, title, description }: MethodCategoryProps) {
  return (
    <div className="flex items-start space-x-4 p-4 bg-[var(--background)] border border-[var(--border-light)] rounded-lg hover:border-[var(--border)] transition-colors">
      <div className={`w-3 h-3 ${color} rounded-full mt-1`}></div>
      <div>
        <div className="font-semibold text-[var(--foreground)] text-sm">
          {title}
        </div>
        <div className="text-xs text-[var(--toss-light-gray)] mt-1">
          {description}
        </div>
      </div>
    </div>
  );
}

export default function MethodsSection() {
  const methods = [
    {
      color: "bg-[var(--toss-blue)]",
      title: "Traditional Methods",
      description:
        "Classical alignment algorithms like Needleman-Wunsch and HHAlign using substitution matrices",
    },
    {
      color: "bg-purple-500",
      title: "OTalign Methods",
      description:
        "Novel optimal transport-based alignment using protein language model embeddings (ESM-2, ESM-1b, ProtT5, AnkhCL)",
    },
    {
      color: "bg-green-500",
      title: "PLM-based Methods",
      description:
        "Methods leveraging protein language models for sequence representation without optimal transport",
    },
  ];

  return (
    <div className="bg-[var(--background)] border border-[var(--border)] rounded-xl p-6">
      <h3 className="font-bold text-[var(--foreground)] text-lg mb-4 flex items-center">
        <span className="mr-2">🔬</span>
        Method Categories
      </h3>
      <div className="space-y-4">
        {methods.map((method) => (
          <MethodCategory key={method.title} {...method} />
        ))}
      </div>
    </div>
  );
}

