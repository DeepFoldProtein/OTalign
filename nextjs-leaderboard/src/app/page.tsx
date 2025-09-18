'use client';

import { useState, useEffect } from 'react';
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from '@headlessui/react';
import { LeaderboardData } from '@/lib/types';
import LeaderboardTable from '@/components/LeaderboardTable';
import PerformanceChart from '@/components/PerformanceChart';
import SubmissionForm from '@/components/SubmissionForm';
import clsx from 'clsx';

export default function Home() {
  const [data, setData] = useState<LeaderboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/leaderboard')
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(() => {
        setError('Failed to load leaderboard data');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading leaderboard...</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">⚠️</div>
          <p className="text-red-600 dark:text-red-400">{error || 'Failed to load data'}</p>
        </div>
      </div>
    );
  }

  const tabs = [
    { name: 'Leaderboard', key: 'leaderboard' },
    { name: 'Analysis', key: 'analysis' },
    { name: 'About', key: 'about' },
    { name: 'Submit', key: 'submit' },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-900 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">OT</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  OTalign Leaderboard
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Protein Sequence Alignment Benchmark
                </p>
              </div>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Last updated: {new Date(data.metadata.last_updated).toLocaleDateString()}
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <div className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-blue-900/20 dark:to-purple-900/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-6">
              Benchmarking Protein Sequence Alignment Methods
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto mb-12">
              Evaluating alignment methods on challenging datasets using Optimal Transport theory
              for comprehensive protein sequence analysis.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mb-4 mx-auto">
                  <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">MALIDUP</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">True homologs with low sequence identity</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                <div className="w-12 h-12 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center mb-4 mx-auto">
                  <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">MALISAM</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Non-homologous structural analogs</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center mb-4 mx-auto">
                  <svg className="w-6 h-6 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">SABmark</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Remote homologs from SCOP superfamilies</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <TabGroup>
          <TabList className="flex space-x-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg mb-8 max-w-md mx-auto">
            {tabs.map((tab) => (
              <Tab
                key={tab.key}
                className={({ selected }) =>
                  clsx(
                    'w-full rounded-md py-3 px-4 text-sm font-semibold leading-5 transition-all',
                    'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
                    selected
                      ? 'bg-white text-gray-900 shadow-sm dark:bg-gray-700 dark:text-white'
                      : 'text-gray-600 hover:bg-white/50 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-700/50 dark:hover:text-white'
                  )
                }
              >
                {tab.name}
              </Tab>
            ))}
          </TabList>
          
          <TabPanels>
            {/* Leaderboard Tab */}
            <TabPanel className="focus:outline-none">
              <div className="space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    Main Leaderboard
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-6">
                    Compare protein alignment methods across different benchmark datasets. 
                    Higher F1 scores and recall values indicate better performance.
                  </p>
                </div>
                <LeaderboardTable data={data.leaderboard_data} />
              </div>
            </TabPanel>

            {/* Analysis Tab */}
            <TabPanel className="focus:outline-none">
              <div className="space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    Performance Analysis
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-6">
                    Visualize and compare method performance across different metrics and datasets.
                  </p>
                </div>
                <PerformanceChart data={data.leaderboard_data} />
              </div>
            </TabPanel>

            {/* About Tab */}
            <TabPanel className="focus:outline-none">
              <div className="max-w-4xl space-y-8">
                <div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
                    About This Leaderboard
                  </h3>
                </div>

                <div className="bg-white dark:bg-gray-800 p-8 rounded-lg border border-gray-200 dark:border-gray-700">
                  <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                    🎯 Evaluation Datasets
                  </h4>
                  
                  <div className="space-y-6">
                    <div>
                      <h5 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">
                        1. MALIDUP (True Homologs)
                      </h5>
                      <p className="text-gray-600 dark:text-gray-400">
                        Contains protein pairs that are true homologs with low sequence identity, often resulting 
                        from domain duplication events. High recall is desired as these represent evolutionarily 
                        related proteins that alignment methods should detect.
                      </p>
                    </div>

                    <div>
                      <h5 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">
                        2. MALISAM (Structural Analogs)
                      </h5>
                      <p className="text-gray-600 dark:text-gray-400">
                        Contains protein pairs that are structural analogs but not evolutionarily related. 
                        Low false positive rate is desired as these represent convergent evolution cases 
                        that should be distinguished from true homology.
                      </p>
                    </div>

                    <div>
                      <h5 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">
                        3. SABmark (Remote Homologs)
                      </h5>
                      <p className="text-gray-600 dark:text-gray-400">
                        Contains challenging cases of remote homologs grouped by SCOP superfamilies, 
                        with structural alignments as ground truth. Tests the ability to detect 
                        distant evolutionary relationships.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 p-8 rounded-lg border border-gray-200 dark:border-gray-700">
                  <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                    📊 Metrics
                  </h4>
                  
                  <div className="space-y-4">
                    <div>
                      <strong className="text-gray-900 dark:text-white">F1 Score:</strong>
                      <span className="text-gray-600 dark:text-gray-400 ml-2">
                        Harmonic mean of precision and recall, providing a balanced measure of performance
                      </span>
                    </div>
                    <div>
                      <strong className="text-gray-900 dark:text-white">Recall:</strong>
                      <span className="text-gray-600 dark:text-gray-400 ml-2">
                        Fraction of true alignments recovered by the method
                      </span>
                    </div>
                    <div>
                      <strong className="text-gray-900 dark:text-white">Precision:</strong>
                      <span className="text-gray-600 dark:text-gray-400 ml-2">
                        Fraction of predicted alignments that are correct
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 p-8 rounded-lg border border-gray-200 dark:border-gray-700">
                  <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                    🔬 Methods
                  </h4>
                  
                  <div className="space-y-4">
                    <div>
                      <strong className="text-blue-600 dark:text-blue-400">Traditional:</strong>
                      <span className="text-gray-600 dark:text-gray-400 ml-2">
                        Classical alignment algorithms like Needleman-Wunsch and HHAlign using 
                        substitution matrices and profile-based approaches
                      </span>
                    </div>
                    <div>
                      <strong className="text-purple-600 dark:text-purple-400">OTalign:</strong>
                      <span className="text-gray-600 dark:text-gray-400 ml-2">
                        Novel optimal transport-based alignment methods using various protein 
                        language model embeddings (ESM-2, ESM-1b, ProtT5, AnkhCL)
                      </span>
                    </div>
                    <div>
                      <strong className="text-green-600 dark:text-green-400">PLM-based:</strong>
                      <span className="text-gray-600 dark:text-gray-400 ml-2">
                        Methods leveraging protein language models for sequence representation 
                        and alignment without optimal transport
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 p-8 rounded-lg border border-blue-200 dark:border-blue-800">
                  <h4 className="text-xl font-semibold text-blue-900 dark:text-blue-100 mb-4">
                    📚 Citation
                  </h4>
                  <pre className="bg-white dark:bg-gray-800 p-4 rounded-md overflow-x-auto text-sm font-mono border border-gray-200 dark:border-gray-600">
                    <code className="text-gray-900 dark:text-gray-100">{`@software{otalign2025,
  title={OTalign: Differentiable OT-based alignment and evaluation for protein embeddings},
  author={Minsoo Kim, Hanjin Bae},
  year={2025}
}`}</code>
                  </pre>
                </div>
              </div>
            </TabPanel>

            {/* Submit Tab */}
            <TabPanel className="focus:outline-none">
              <SubmissionForm />
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600 dark:text-gray-400">
            <p className="mb-2">
              OTalign Leaderboard | Total models: {data.metadata.total_models} | 
              Version: {data.metadata.version}
            </p>
            <div className="flex justify-center space-x-6 text-sm">
              <a 
                href="https://github.com/your-username/OTalign" 
                className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub Repository
              </a>
              <span>Contact: your-email@domain.com</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
