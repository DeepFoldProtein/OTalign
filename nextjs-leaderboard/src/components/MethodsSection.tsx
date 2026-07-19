interface MethodCategoryProps {
  color: string;
  title: string;
  description: string;
}

function MethodCategory({ color, title, description }: MethodCategoryProps) {
  return (
    <div className="flex items-start gap-3 rounded-[var(--r-ctrl)] border border-[var(--line)] bg-[var(--surface-2)] p-4">
      <span
        className="w-2.5 h-2.5 rounded-full mt-1.5 shrink-0"
        style={{ backgroundColor: color }}
      />
      <div>
        <div className="font-semibold text-[var(--ink)] text-[14px]">
          {title}
        </div>
        <div className="text-[12.5px] text-[var(--ink-2)] mt-0.5 leading-relaxed">
          {description}
        </div>
      </div>
    </div>
  );
}

export default function MethodsSection() {
  const methods = [
    {
      color: "var(--series-ot)",
      title: "OTalign methods",
      description:
        "Optimal-transport alignment over protein language model embeddings (ESM-2, ESM-1b, ProtT5, Ankh).",
    },
    {
      color: "var(--series-plm)",
      title: "PLM-based methods",
      description:
        "Methods leveraging protein language model representations without optimal transport.",
    },
    {
      color: "var(--series-trad)",
      title: "Traditional methods",
      description:
        "Classical algorithms such as Needleman-Wunsch and HHAlign using substitution matrices.",
    },
  ];

  return (
    <div className="card p-5 sm:p-6">
      <h3 className="font-semibold text-[var(--ink)] text-[15px] mb-4">
        Method categories
      </h3>
      <div className="space-y-3">
        {methods.map((m) => (
          <MethodCategory key={m.title} {...m} />
        ))}
      </div>
    </div>
  );
}
