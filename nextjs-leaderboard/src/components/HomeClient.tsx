"use client";

import Image from "next/image";
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@headlessui/react";
import { useLeaderboard } from "@/hooks/useLeaderboard";
import LeaderboardTable from "@/components/LeaderboardTable";
import PerformanceChart from "@/components/PerformanceChart";
import SubmissionForm from "@/components/SubmissionForm";
import DatasetCards from "@/components/DatasetCards";
import MetricsSection from "@/components/MetricsSection";
import MethodsSection from "@/components/MethodsSection";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorDisplay from "@/components/ErrorDisplay";
import clsx from "clsx";

export default function HomeClient() {
  const { data, loading, error } = useLeaderboard();

  if (loading) {
    return <LoadingSpinner message="Loading leaderboard…" />;
  }

  if (error || !data) {
    return <ErrorDisplay message={error || "Failed to load data"} />;
  }

  const tabs = [
    { name: "Leaderboard", key: "leaderboard" },
    { name: "Analysis", key: "analysis" },
    { name: "Datasets", key: "datasets" },
    { name: "Submit", key: "submit" },
  ];

  return (
    <div className="min-h-screen bg-[var(--page)] flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-[var(--line)] bg-[var(--surface)]/80 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-2.5">
              <Image
                src="/svgs/foreground.svg"
                alt="OTalign"
                width={26}
                height={26}
                className="w-[26px] h-[26px]"
              />
              <span className="font-semibold text-[var(--ink)] text-[15px] tracking-tight">
                OTalign
              </span>
              <span className="hidden sm:inline text-[13px] text-[var(--ink-3)] border-l border-[var(--line-2)] pl-2.5">
                Protein Alignment Benchmark
              </span>
            </div>

            <a
              href="https://github.com/DeepFoldProtein/OTalign"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[13px] text-[var(--ink-2)] hover:text-[var(--ink)] transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="border-b border-[var(--line)] bg-[var(--surface)]">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 lg:px-8 pt-10 pb-9">
          <p className="text-[13px] font-medium text-[var(--accent)] mb-2">
            Optimal Transport · Protein Alignment
          </p>
          <h1 className="text-[28px] sm:text-[34px] font-bold tracking-tight text-[var(--ink)] max-w-2xl leading-[1.15]">
            Benchmarking protein sequence alignment methods
          </h1>
          <p className="mt-3 text-[15px] text-[var(--ink-2)] max-w-2xl leading-relaxed">
            A living leaderboard comparing optimal-transport, protein language
            model, and classical alignment methods across four challenging
            structural benchmarks.
          </p>
        </div>
      </section>

      {/* Main */}
      <main className="max-w-6xl mx-auto w-full px-5 sm:px-6 lg:px-8 py-7 flex-grow">
        <TabGroup>
          <TabList className="flex gap-1 mb-7 border-b border-[var(--line)]">
            {tabs.map((tab) => (
              <Tab
                key={tab.key}
                className={({ selected }) =>
                  clsx(
                    "relative -mb-px px-3.5 py-2.5 text-[14px] font-medium transition-colors focus:outline-none",
                    selected
                      ? "text-[var(--ink)] border-b-2 border-[var(--accent)]"
                      : "text-[var(--ink-3)] border-b-2 border-transparent hover:text-[var(--ink-2)]"
                  )
                }
              >
                {tab.name}
              </Tab>
            ))}
          </TabList>

          <TabPanels>
            <TabPanel className="focus:outline-none">
              <SectionHeader
                title="Leaderboard"
                subtitle="Ranked by average score across MALIDUP, MALISAM, and SABmark benchmarks."
              />
              <LeaderboardTable data={data.leaderboard_data} />
            </TabPanel>

            <TabPanel className="focus:outline-none">
              <SectionHeader
                title="Performance analysis"
                subtitle="Method performance across parameter scale and per-benchmark profiles."
              />
              <PerformanceChart data={data.leaderboard_data} />
            </TabPanel>

            <TabPanel className="focus:outline-none">
              <SectionHeader
                title="Benchmark datasets"
                subtitle="Evaluation sets probing distinct alignment challenges."
              />
              <div className="space-y-6">
                <DatasetCards />
                <MetricsSection />
                <MethodsSection />
              </div>
            </TabPanel>

            <TabPanel className="focus:outline-none">
              <SectionHeader
                title="Submit your method"
                subtitle="Generate a submission entry and open a pull request to add it."
              />
              <SubmissionForm />
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--line)] bg-[var(--surface)]">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 lg:px-8 py-5">
          <div className="flex items-center justify-between text-[13px] text-[var(--ink-3)]">
            <div className="flex items-center gap-3">
              <span>OTalign v{data.metadata.version}</span>
              <span className="text-[var(--line-2)]">·</span>
              <span>
                Updated{" "}
                {new Date(data.metadata.last_updated).toLocaleDateString()}
              </span>
            </div>
            <span>© 2025 DeepFold</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

function SectionHeader({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  return (
    <div className="mb-5">
      <h2 className="text-[20px] font-bold tracking-tight text-[var(--ink)]">
        {title}
      </h2>
      <p className="text-[14px] text-[var(--ink-2)] mt-1">{subtitle}</p>
    </div>
  );
}
