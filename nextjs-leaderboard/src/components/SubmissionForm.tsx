"use client";

import { useSubmissionForm } from "@/hooks/useSubmissionForm";

const inputCls =
  "w-full rounded-[var(--r-ctrl)] border border-[var(--line-2)] bg-[var(--surface)] px-3 py-2 text-[14px] text-[var(--ink)] placeholder:text-[var(--ink-3)] focus:outline-none focus:border-[var(--accent)] transition-colors";
const labelCls =
  "block text-[13px] font-medium text-[var(--ink-2)] mb-1.5";

export default function SubmissionForm() {
  const {
    formData,
    generatedJson,
    handleInputChange,
    generateSubmission,
    copyToClipboard,
    isFormValid,
  } = useSubmissionForm();

  const metrics = [
    { id: "malidup_f1", label: "MALIDUP F1" },
    { id: "malisam_f1", label: "MALISAM F1" },
    { id: "sabmark_sup_recall", label: "SABmark (sup) recall" },
    { id: "sabmark_twi_recall", label: "SABmark (twi) recall" },
  ] as const;

  return (
    <div className="max-w-3xl space-y-5">
      {/* Steps */}
      <div className="rounded-[var(--r-card)] border border-[var(--accent)]/25 bg-[var(--accent-weak)] p-5">
        <h3 className="text-[15px] font-semibold text-[var(--ink)] mb-3">
          How to submit
        </h3>
        <ol className="space-y-2 text-[13.5px] text-[var(--ink-2)]">
          {[
            ["Run evaluation", "Execute your alignment method on our benchmark datasets."],
            ["Fill the form", "Enter your method details and per-benchmark scores below."],
            ["Generate JSON", "Produce the submission entry with the button below."],
            ["Open a PR", "Add the entry via a pull request to our GitHub repository."],
          ].map(([t, d], i) => (
            <li key={t} className="flex gap-2.5">
              <span className="tnum shrink-0 w-5 h-5 rounded-full bg-[var(--accent)] text-[var(--accent-ink)] text-[11px] font-semibold flex items-center justify-center">
                {i + 1}
              </span>
              <span>
                <strong className="text-[var(--ink)] font-semibold">{t}.</strong>{" "}
                {d}
              </span>
            </li>
          ))}
        </ol>
      </div>

      {/* Form */}
      <div className="card p-5 sm:p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="space-y-4">
            <h3 className="text-[14px] font-semibold text-[var(--ink)]">
              Basic information
            </h3>
            <div>
              <label htmlFor="model" className={labelCls}>
                Method name *
              </label>
              <input
                type="text"
                id="model"
                name="model"
                value={formData.model}
                onChange={handleInputChange}
                placeholder="YourMethod-v1.0"
                required
                className={inputCls}
              />
            </div>
            <div>
              <label htmlFor="type" className={labelCls}>
                Method type *
              </label>
              <select
                id="type"
                name="type"
                value={formData.type}
                onChange={handleInputChange}
                required
                className={inputCls}
              >
                <option value="">Select type…</option>
                <option value="Traditional">Traditional</option>
                <option value="PLM-based">PLM-based</option>
                <option value="OTalign">OTalign</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div>
              <label htmlFor="organization" className={labelCls}>
                Organization *
              </label>
              <input
                type="text"
                id="organization"
                name="organization"
                value={formData.organization}
                onChange={handleInputChange}
                placeholder="Your university or company"
                required
                className={inputCls}
              />
            </div>
            <div>
              <label htmlFor="description" className={labelCls}>
                Description *
              </label>
              <textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Brief description of your method…"
                rows={3}
                required
                className={inputCls}
              />
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-[14px] font-semibold text-[var(--ink)]">
              Links & performance
            </h3>
            <div>
              <label htmlFor="code_url" className={labelCls}>
                Code URL *
              </label>
              <input
                type="url"
                id="code_url"
                name="code_url"
                value={formData.code_url}
                onChange={handleInputChange}
                placeholder="https://github.com/…"
                required
                className={inputCls}
              />
            </div>
            <div>
              <label htmlFor="paper_url" className={labelCls}>
                Paper URL (optional)
              </label>
              <input
                type="url"
                id="paper_url"
                name="paper_url"
                value={formData.paper_url}
                onChange={handleInputChange}
                placeholder="https://arxiv.org/abs/…"
                className={inputCls}
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              {metrics.map((m) => (
                <div key={m.id}>
                  <label htmlFor={m.id} className="block text-[12px] font-medium text-[var(--ink-2)] mb-1.5">
                    {m.label}
                  </label>
                  <input
                    type="number"
                    id={m.id}
                    name={m.id}
                    value={(formData[m.id] as number | undefined) || ""}
                    onChange={handleInputChange}
                    step="0.0001"
                    min="0"
                    max="1"
                    placeholder="0.0000"
                    className={`${inputCls} tnum`}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        <button
          onClick={generateSubmission}
          disabled={!isFormValid}
          className="mt-6 w-full rounded-[var(--r-ctrl)] bg-[var(--accent)] text-[var(--accent-ink)] py-2.5 text-[14px] font-semibold hover:opacity-90 focus:outline-none disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
        >
          Generate submission JSON
        </button>
      </div>

      {generatedJson && (
        <div className="card p-5 sm:p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[15px] font-semibold text-[var(--ink)]">
              Generated submission
            </h3>
            <button
              onClick={copyToClipboard}
              className="rounded-[var(--r-ctrl)] border border-[var(--line-2)] bg-[var(--surface-2)] px-3 py-1.5 text-[13px] font-medium text-[var(--ink)] hover:bg-[var(--surface-hover)] transition-colors"
            >
              Copy
            </button>
          </div>
          <pre className="rounded-[var(--r-ctrl)] border border-[var(--line)] bg-[var(--surface-2)] p-4 overflow-x-auto text-[12.5px] font-mono text-[var(--ink)]">
            <code>{generatedJson}</code>
          </pre>
          <p className="mt-4 text-[13px] text-[var(--ink-2)]">
            Save this to a file and submit it via a pull request to{" "}
            <a
              href="https://github.com/DeepFoldProtein/OTalign"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[var(--accent)] hover:underline"
            >
              our GitHub repository
            </a>
            .
          </p>
        </div>
      )}
    </div>
  );
}
