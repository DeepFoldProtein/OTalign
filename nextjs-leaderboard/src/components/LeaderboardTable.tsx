"use client";

import { LeaderboardEntry } from "@/lib/types";
import { methodColorVar } from "@/lib/methodTypes";
import { ChevronUpIcon, ChevronDownIcon } from "@heroicons/react/24/solid";
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

  const maxAvg = Math.max(
    ...data.map((d) => d.average ?? 0),
    0.0001
  );

  const filters = ["all", ...uniqueTypes];

  const formatScore = (score: number | null) =>
    score === null ? (
      <span className="text-[var(--ink-3)]">—</span>
    ) : (
      <span className="tnum">{score.toFixed(4)}</span>
    );

  return (
    <div className="space-y-4">
      {/* Filter row */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex items-center gap-1 rounded-[var(--r-ctrl)] border border-[var(--line)] bg-[var(--surface-2)] p-1">
          {filters.map((type) => (
            <button
              key={type}
              onClick={() => setTypeFilter(type)}
              className={clsx(
                "px-3 py-1.5 text-[13px] font-medium rounded-md transition-colors focus:outline-none",
                typeFilter === type
                  ? "bg-[var(--surface)] text-[var(--ink)] shadow-sm border border-[var(--line-2)]"
                  : "text-[var(--ink-3)] hover:text-[var(--ink)]"
              )}
            >
              {type === "all" ? "All methods" : type}
            </button>
          ))}
        </div>
        <span className="text-[13px] text-[var(--ink-3)]">
          {sortedAndFilteredData.length} of {data.length} methods
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-[var(--r-card)] border border-[var(--line)] bg-[var(--surface)]">
        <table className="w-full text-[13.5px]">
          <thead>
            <tr className="border-b border-[var(--line)]">
              <SortableTh
                label="#"
                field="rank"
                {...{ sortField, sortDirection, handleSort }}
                className="w-14"
              />
              <SortableTh
                label="Method"
                field="model"
                {...{ sortField, sortDirection, handleSort }}
              />
              <SortableTh
                label="Type"
                field="type"
                {...{ sortField, sortDirection, handleSort }}
              />
              <SortableTh
                label="Params"
                field="parameters"
                align="right"
                {...{ sortField, sortDirection, handleSort }}
              />
              <SortableTh
                label="Avg."
                field="average"
                align="right"
                {...{ sortField, sortDirection, handleSort }}
                className="min-w-[150px]"
              />
              <SortableTh
                label="SABmark sup"
                field="sabmark_sup_recall"
                align="right"
                {...{ sortField, sortDirection, handleSort }}
              />
              <SortableTh
                label="SABmark twi"
                field="sabmark_twi_recall"
                align="right"
                {...{ sortField, sortDirection, handleSort }}
              />
              <SortableTh
                label="MALIDUP F1"
                field="malidup_f1"
                align="right"
                {...{ sortField, sortDirection, handleSort }}
              />
              <SortableTh
                label="MALISAM F1"
                field="malisam_f1"
                align="right"
                {...{ sortField, sortDirection, handleSort }}
              />
              <Th label="Organization" />
              <Th label="Links" />
            </tr>
          </thead>
          <tbody>
            {sortedAndFilteredData.map((entry) => {
              const isRanked = sortField === "rank" && sortDirection === "asc";
              return (
                <tr
                  key={entry.model}
                  className="border-b border-[var(--line)] last:border-0 hover:bg-[var(--surface-2)] transition-colors"
                >
                  {/* Rank */}
                  <td className="px-4 py-3 tnum text-[var(--ink-2)]">
                    <span
                      className={clsx(
                        isRanked &&
                          entry.rank <= 3 &&
                          "font-semibold text-[var(--ink)]"
                      )}
                    >
                      {entry.rank}
                    </span>
                  </td>

                  {/* Method */}
                  <td className="px-4 py-3">
                    <div className="font-medium text-[var(--ink)] whitespace-nowrap">
                      {entry.model}
                    </div>
                    <div
                      className="text-[12px] text-[var(--ink-3)] max-w-[280px] truncate"
                      title={entry.description}
                    >
                      {entry.description}
                    </div>
                  </td>

                  {/* Type */}
                  <td className="px-4 py-3">
                    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-[var(--ink-2)]">
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: methodColorVar(entry.type) }}
                      />
                      {entry.type}
                    </span>
                  </td>

                  {/* Params */}
                  <td className="px-4 py-3 text-right tnum text-[var(--ink-2)] whitespace-nowrap">
                    {entry.parameters}
                  </td>

                  {/* Avg + magnitude bar */}
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2.5 justify-end">
                      <div
                        className="hidden sm:block h-1.5 w-16 rounded-full overflow-hidden shrink-0"
                        style={{ background: "var(--bar-track)" }}
                      >
                        {entry.average !== null && (
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${(entry.average / maxAvg) * 100}%`,
                              background: "var(--bar-fill)",
                            }}
                          />
                        )}
                      </div>
                      <span className="font-semibold text-[var(--ink)] tabular-nums w-[52px] text-right">
                        {formatScore(entry.average)}
                      </span>
                    </div>
                  </td>

                  <td className="px-4 py-3 text-right text-[var(--ink-2)]">
                    {formatScore(entry.sabmark_sup_recall)}
                  </td>
                  <td className="px-4 py-3 text-right text-[var(--ink-2)]">
                    {formatScore(entry.sabmark_twi_recall)}
                  </td>
                  <td className="px-4 py-3 text-right text-[var(--ink-2)]">
                    {formatScore(entry.malidup_f1)}
                  </td>
                  <td className="px-4 py-3 text-right text-[var(--ink-2)]">
                    {formatScore(entry.malisam_f1)}
                  </td>

                  <td className="px-4 py-3 text-[var(--ink-2)] whitespace-nowrap">
                    {entry.organization}
                  </td>

                  {/* Links */}
                  <td className="px-4 py-3 whitespace-nowrap">
                    <div className="flex items-center gap-3">
                      {entry.paper_url && (
                        <a
                          href={entry.paper_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[13px] text-[var(--accent)] hover:underline"
                        >
                          Paper
                        </a>
                      )}
                      {entry.code_url && (
                        <a
                          href={entry.code_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[13px] text-[var(--ink-2)] hover:text-[var(--ink)] hover:underline"
                        >
                          Code
                        </a>
                      )}
                      {!entry.paper_url && !entry.code_url && (
                        <span className="text-[var(--ink-3)]">—</span>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Th({ label }: { label: string }) {
  return (
    <th className="px-4 py-3 text-left text-[12px] font-semibold text-[var(--ink-3)] whitespace-nowrap">
      {label}
    </th>
  );
}

function SortableTh({
  label,
  field,
  align = "left",
  className,
  sortField,
  sortDirection,
  handleSort,
}: {
  label: string;
  field: SortField;
  align?: "left" | "right";
  className?: string;
  sortField: SortField;
  sortDirection: "asc" | "desc";
  handleSort: (f: SortField) => void;
}) {
  const active = sortField === field;
  return (
    <th
      onClick={() => handleSort(field)}
      className={clsx(
        "px-4 py-3 text-[12px] font-semibold whitespace-nowrap cursor-pointer select-none transition-colors",
        active ? "text-[var(--ink)]" : "text-[var(--ink-3)] hover:text-[var(--ink-2)]",
        className
      )}
    >
      <div
        className={clsx(
          "flex items-center gap-1",
          align === "right" && "justify-end"
        )}
      >
        {label}
        <span className="w-3">
          {active &&
            (sortDirection === "asc" ? (
              <ChevronUpIcon className="w-3 h-3" />
            ) : (
              <ChevronDownIcon className="w-3 h-3" />
            ))}
        </span>
      </div>
    </th>
  );
}
