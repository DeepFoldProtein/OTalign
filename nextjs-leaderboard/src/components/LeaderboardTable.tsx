"use client";

import { LeaderboardEntry } from "@/lib/types";
import {
  ChevronUpIcon,
  ChevronDownIcon,
  LinkIcon,
} from "@heroicons/react/24/outline";
import clsx from "clsx";
import { useTableSort, SortField } from "@/hooks/useTableSort";

interface LeaderboardTableProps {
  data: LeaderboardEntry[];
}

export default function LeaderboardTable({ data }: LeaderboardTableProps) {
  const {
    sortField,
    sortDirection,
    typeFilter,
    sortedAndFilteredData,
    uniqueTypes,
    handleSort,
    setTypeFilter,
  } = useTableSort({ data });

  const formatScore = (score: number | null) => {
    if (score === null) return <span className="text-gray-400">TBD</span>;
    return <span className="font-mono">{score.toFixed(4)}</span>;
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "Traditional":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300";
      case "OTalign":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300";
      case "PLM-Based":
      case "PLM-based":
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300";
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDirection === "asc" ? (
      <ChevronUpIcon className="w-4 h-4" />
    ) : (
      <ChevronDownIcon className="w-4 h-4" />
    );
  };

  return (
    <div className="space-y-4">
      {/* Filter Controls */}
      <div className="flex flex-wrap gap-4 items-center justify-between">
        <div className="flex items-center gap-2">
          <label
            htmlFor="type-filter"
            className="text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Filter by type:
          </label>
          <select
            id="type-filter"
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="rounded-md border border-gray-300 bg-white px-3 py-1 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-800 dark:text-white"
          >
            <option value="all">All Types</option>
            {uniqueTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>
        <div className="text-sm text-gray-600 dark:text-gray-400">
          Showing {sortedAndFilteredData.length} of {data.length} entries
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
        <table className="w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("rank")}
              >
                <div className="flex items-center gap-1">
                  Rank
                  <SortIcon field="rank" />
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("model")}
              >
                <div className="flex items-center gap-1">
                  Model
                  <SortIcon field="model" />
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("type")}
              >
                <div className="flex items-center gap-1">
                  Type
                  <SortIcon field="type" />
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("parameters")}
              >
                <div className="flex items-center gap-1">
                  Parameters
                  <SortIcon field="parameters" />
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("average")}
              >
                <div className="flex items-center gap-1">
                  Avg
                  <SortIcon field="average" />
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("malidup_f1")}
              >
                <div className="flex items-center gap-1">
                  MALIDUP F1
                  <SortIcon field="malidup_f1" />
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("malisam_f1")}
              >
                <div className="flex items-center gap-1">
                  MALISAM F1
                  <SortIcon field="malisam_f1" />
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("sabmark_sup_recall")}
              >
                <div className="flex items-center gap-1">
                  SABmark (sup) Recall
                  <SortIcon field="sabmark_sup_recall" />
                </div>
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("sabmark_twi_recall")}
              >
                <div className="flex items-center gap-1">
                  SABmark (twi) Recall
                  <SortIcon field="sabmark_twi_recall" />
                </div>
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-400">
                Organization
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700"
                onClick={() => handleSort("date_submitted")}
              >
                <div className="flex items-center gap-1">
                  Date
                  <SortIcon field="date_submitted" />
                </div>
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-400">
                Links
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-400">
                Description
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-900 dark:divide-gray-700">
            {sortedAndFilteredData.map((entry, index) => (
              <tr
                key={entry.model}
                className={clsx(
                  "hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors",
                  index < 3 &&
                    sortField === "rank" &&
                    sortDirection === "asc" &&
                    "bg-gradient-to-r from-yellow-50 to-transparent dark:from-yellow-900/20"
                )}
              >
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    {entry.rank <= 3 &&
                      sortField === "rank" &&
                      sortDirection === "asc" && (
                        <span className="mr-2">
                          {entry.rank === 1
                            ? "🥇"
                            : entry.rank === 2
                            ? "🥈"
                            : "🥉"}
                        </span>
                      )}
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {entry.rank}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-semibold text-gray-900 dark:text-white">
                    {entry.model}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span
                    className={clsx(
                      "inline-flex px-2 py-1 text-xs font-medium rounded-full",
                      getTypeColor(entry.type)
                    )}
                  >
                    {entry.type}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  <span className="font-mono">{entry.parameters}</span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {formatScore(entry.average)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {formatScore(entry.malidup_f1)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {formatScore(entry.malisam_f1)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {formatScore(entry.sabmark_sup_recall)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {formatScore(entry.sabmark_twi_recall)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {entry.organization}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {entry.date_submitted}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <div className="flex space-x-3">
                    {entry.paper_url && (
                      <a
                        href={entry.paper_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center px-2 py-1 text-xs font-medium text-blue-700 bg-blue-100 rounded hover:bg-blue-200 dark:text-blue-300 dark:bg-blue-900/30 dark:hover:bg-blue-900/50 transition-colors"
                        title="Paper"
                      >
                        Paper
                      </a>
                    )}
                    {entry.code_url && (
                      <a
                        href={entry.code_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center px-2 py-1 text-xs font-medium text-gray-700 bg-gray-100 rounded hover:bg-gray-200 dark:text-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 transition-colors"
                        title="Code"
                      >
                        <LinkIcon className="w-3 h-3 mr-1" />
                        Code
                      </a>
                    )}
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div
                    className="text-sm text-gray-900 dark:text-white max-w-xs truncate"
                    title={entry.description}
                  >
                    {entry.description}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
