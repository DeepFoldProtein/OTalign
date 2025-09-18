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
    return <LoadingSpinner message="Loading leaderboard..." />;
  }

  if (error || !data) {
    return <ErrorDisplay message={error || "Failed to load data"} />;
  }

  const tabs = [
    { name: "Leaderboard", key: "leaderboard", icon: "📊" },
    { name: "Analysis", key: "analysis", icon: "📈" },
    { name: "Datasets", key: "datasets", icon: "🎯" },
    { name: "Submit", key: "submit", icon: "📤" },
  ];

  return (
    <div className="min-h-screen bg-[var(--background)] flex flex-col">
      {/* Compact Header */}
      <header className="bg-[var(--background)] border-b border-[var(--border-light)] sticky top-0 z-50 backdrop-blur-sm bg-opacity-95">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-8">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8">
                  <Image
                    src="/svgs/foreground.svg"
                    alt="OTalign"
                    width={32}
                    height={32}
                    className="w-full h-full"
                  />
                </div>
                <div>
                  <h1 className="font-bold text-[var(--foreground)] text-lg">
                    OTalign
                  </h1>
                  <p className="text-xs text-[var(--toss-light-gray)] leading-tight">
                    Protein Alignment Benchmark
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-xs text-[var(--toss-light-gray)]">
                {data.metadata.total_models} models
              </div>
              <div className="text-xs text-[var(--toss-light-gray)]">
                Updated{" "}
                {new Date(data.metadata.last_updated).toLocaleDateString()}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-grow">
        <TabGroup>
          <TabList className="flex space-x-1 bg-[var(--surface)] border border-[var(--border-light)] p-1 rounded-lg mb-6 max-w-lg mx-auto">
            {tabs.map((tab) => (
              <Tab
                key={tab.key}
                className={({ selected }) =>
                  clsx(
                    "flex-1 rounded-md py-2 px-3 text-sm font-medium leading-5 transition-all flex items-center justify-center space-x-2",
                    "focus:outline-none focus:ring-2 focus:ring-[var(--toss-blue)] focus:ring-offset-1",
                    selected
                      ? "bg-[var(--background)] text-[var(--foreground)] shadow-sm border border-[var(--border)]"
                      : "text-[var(--toss-light-gray)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)]"
                  )
                }
              >
                <span className="text-xs">{tab.icon}</span>
                <span>{tab.name}</span>
              </Tab>
            ))}
          </TabList>

          <TabPanels>
            {/* Leaderboard Tab */}
            <TabPanel className="focus:outline-none">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-bold text-[var(--foreground)]">
                      Protein Alignment Leaderboard
                    </h2>
                    <p className="text-sm text-[var(--toss-light-gray)] mt-1">
                      Comparing alignment methods across challenging benchmark
                      datasets
                    </p>
                  </div>
                  <div className="text-xs text-[var(--toss-light-gray)]">
                    Sorted by F1 Score
                  </div>
                </div>
                <LeaderboardTable data={data.leaderboard_data} />
              </div>
            </TabPanel>

            {/* Analysis Tab */}
            <TabPanel className="focus:outline-none">
              <div className="space-y-4">
                <div>
                  <h2 className="text-xl font-bold text-[var(--foreground)]">
                    Performance Analysis
                  </h2>
                  <p className="text-sm text-[var(--toss-light-gray)] mt-1">
                    Visualize method performance across metrics and datasets
                  </p>
                </div>
                <PerformanceChart data={data.leaderboard_data} />
              </div>
            </TabPanel>

            {/* Datasets Tab */}
            <TabPanel className="focus:outline-none">
              <div className="space-y-6">
                <div>
                  <h2 className="text-xl font-bold text-[var(--foreground)]">
                    Benchmark Datasets
                  </h2>
                  <p className="text-sm text-[var(--toss-light-gray)] mt-1">
                    Evaluation datasets for protein sequence alignment methods
                    using Optimal Transport theory
                  </p>
                </div>

                {/* Dataset Cards */}
                <DatasetCards />

                {/* Metrics Section */}
                <MetricsSection />

                {/* Methods Overview */}
                <MethodsSection />
              </div>
            </TabPanel>

            {/* Submit Tab */}
            <TabPanel className="focus:outline-none">
              <div className="space-y-4">
                <div>
                  <h2 className="text-xl font-bold text-[var(--foreground)]">
                    Submit Results
                  </h2>
                  <p className="text-sm text-[var(--toss-light-gray)] mt-1">
                    Add your method to the leaderboard
                  </p>
                </div>
                <SubmissionForm />
              </div>
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </main>

      {/* Minimal Footer */}
      <footer className="border-t border-[var(--border-light)] mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between text-xs text-[var(--toss-light-gray)]">
            <div className="flex items-center space-x-4">
              <span>OTalign v{data.metadata.version}</span>
              <span>•</span>
              <a
                href="https://github.com/DeepFoldProtein/OTalign"
                className="hover:text-[var(--toss-blue)] transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </div>
            <div>© 2025 DeepFold Team</div>
          </div>
        </div>
      </footer>
    </div>
  );
}
