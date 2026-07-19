"use client";

import { LeaderboardEntry } from "@/lib/types";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { usePerformanceChart } from "@/hooks/usePerformanceChart";
import { useThemeColors } from "@/hooks/useThemeColors";
import { ScatterCustomizedShape } from "recharts/types/cartesian/Scatter";

interface PerformanceChartProps {
  data: LeaderboardEntry[];
}

interface ScatterDataPoint {
  x: number;
  y: number;
  model: string;
  type: string;
  organization: string;
  parameters: string;
}

export default function PerformanceChart({ data }: PerformanceChartProps) {
  const { selectedModel, scatterData, availableModels, setSelectedModel } =
    usePerformanceChart({ data });

  // Per-benchmark profile: selected method's scores + best-in-class reference.
  const metricDefs = [
    { label: "MALIDUP F1", key: "malidup_f1" },
    { label: "MALISAM F1", key: "malisam_f1" },
    { label: "SABmark (sup)", key: "sabmark_sup_recall" },
    { label: "SABmark (twi)", key: "sabmark_twi_recall" },
  ] as const;
  const selectedEntry = data.find((d) => d.model === selectedModel);
  const benchmarkRows = metricDefs.map((m) => ({
    label: m.label,
    value: selectedEntry?.[m.key] ?? null,
    best: Math.max(...data.map((d) => d[m.key] ?? 0)),
  }));

  // Resolved hex — recharts renders SVG presentation attributes where var() won't apply.
  const c = useThemeColors();
  const AXIS = c.axis;
  const GRID = c.grid;
  const getTypeColor = (type: string) => {
    switch (type) {
      case "OTalign":
        return c.seriesOt;
      case "PLM-Based":
      case "PLM-based":
        return c.seriesPlm;
      case "Traditional":
        return c.seriesTrad;
      default:
        return c.axis;
    }
  };

  const validParameterData = scatterData.filter((d) => d.x >= 0);
  const naParameterData = scatterData.filter((d) => d.x === -1);

  const formatParameterCount = (value: number) => {
    if (value === -1) return "N/A";
    if (value >= 1e9) return `${(value / 1e9).toFixed(0)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(0)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
    return value.toString();
  };

  const getNALineColor = (modelName: string) =>
    modelName.includes("Needleman-Wunsch") || modelName.includes("HHAlign")
      ? c.seriesTrad
      : c.axis;

  const CustomDot = (props: ScatterCustomizedShape) => {
    const { cx, cy, payload } = props as {
      cx: number;
      cy: number;
      payload: ScatterDataPoint;
    };
    if (!payload) return null;
    const color = getTypeColor(payload.type);
    return (
      <circle
        cx={cx}
        cy={cy}
        r={6}
        fill={color}
        stroke={c.surface}
        strokeWidth={2}
      />
    );
  };

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: Array<{ payload: ScatterDataPoint }>;
  }) => {
    if (!active || !payload?.length) return null;
    const valid = payload.filter(
      (p) => p.payload?.model && p.payload.y !== undefined
    );
    if (!valid.length) return null;
    const d = valid[0].payload;
    return (
      <div className="rounded-[var(--r-ctrl)] border border-[var(--line-2)] bg-[var(--surface)] px-3 py-2 shadow-lg">
        <p className="font-semibold text-[var(--ink)] text-[13px]">{d.model}</p>
        <p className="text-[12px] text-[var(--ink-3)] mb-1">{d.organization}</p>
        <p className="text-[12px] text-[var(--ink-2)]">
          Params <span className="tnum">{d.parameters}</span>
        </p>
        <p className="text-[12px] text-[var(--ink-2)]">
          Avg. score <span className="tnum">{d.y.toFixed(4)}</span>
        </p>
      </div>
    );
  };

  const scatterTypes = Array.from(new Set(validParameterData.map((d) => d.type)));

  return (
    <div className="space-y-5">
      {/* Scatter */}
      <div className="card p-5 sm:p-6">
        <h3 className="text-[16px] font-semibold text-[var(--ink)]">
          Parameter scale vs. average performance
        </h3>
        <p className="text-[13px] text-[var(--ink-2)] mt-1 mb-4">
          Each point is a method; the x-axis is on a log scale. Dashed lines mark
          reference methods with no parameter count.
        </p>

        {scatterData.length > 0 ? (
          <>
            <div className="h-[420px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 16, right: 24, bottom: 32, left: 8 }}
                >
                  <CartesianGrid stroke={GRID} strokeDasharray="0" vertical={false} />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="Parameters"
                    domain={[1000000, 120000000000]}
                    scale="log"
                    tickFormatter={formatParameterCount}
                    ticks={Array.from(
                      new Set(validParameterData.map((d) => d.x))
                    ).sort((a, b) => a - b)}
                    tick={{ fill: AXIS, fontSize: 12 }}
                    tickLine={{ stroke: GRID }}
                    axisLine={{ stroke: GRID }}
                    label={{
                      value: "Parameter count",
                      position: "insideBottom",
                      offset: -18,
                      fill: AXIS,
                      fontSize: 12,
                    }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="Average Score"
                    domain={[0.2, 0.55]}
                    tickFormatter={(v) => v.toFixed(2)}
                    tick={{ fill: AXIS, fontSize: 12 }}
                    tickLine={{ stroke: GRID }}
                    axisLine={{ stroke: GRID }}
                    label={{
                      value: "Average score",
                      angle: -90,
                      position: "insideLeft",
                      fill: AXIS,
                      fontSize: 12,
                      style: { textAnchor: "middle" },
                    }}
                  />
                  <Tooltip
                    content={<CustomTooltip />}
                    animationDuration={0}
                    cursor={{ stroke: GRID }}
                  />
                  {naParameterData.map((item, i) => (
                    <ReferenceLine
                      key={`na-${i}`}
                      y={item.y}
                      stroke={getNALineColor(item.model)}
                      strokeDasharray="4 4"
                      strokeWidth={1.5}
                    />
                  ))}
                  <Scatter
                    data={validParameterData}
                    shape={CustomDot as ScatterCustomizedShape}
                    isAnimationActive={false}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-[var(--line)] pt-4">
              {scatterTypes.map((type) => (
                <span
                  key={type}
                  className="inline-flex items-center gap-1.5 text-[13px] text-[var(--ink-2)]"
                >
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: getTypeColor(type) }}
                  />
                  {type}
                </span>
              ))}
              {naParameterData.map((item, i) => (
                <span
                  key={`lg-${i}`}
                  className="inline-flex items-center gap-1.5 text-[13px] text-[var(--ink-2)]"
                >
                  <span
                    className="w-4 border-t-2 border-dashed"
                    style={{ borderColor: getNALineColor(item.model) }}
                  />
                  {item.model}
                </span>
              ))}
            </div>
          </>
        ) : (
          <EmptyState text="No performance data available yet." />
        )}
      </div>

      {/* Per-benchmark profile */}
      <div className="card p-5 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
          <div>
            <h3 className="text-[16px] font-semibold text-[var(--ink)]">
              Per-benchmark profile
            </h3>
            <p className="text-[13px] text-[var(--ink-2)] mt-1">
              Scores across the four benchmarks. The tick marks the best result
              across all methods.
            </p>
          </div>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="rounded-[var(--r-ctrl)] border border-[var(--line-2)] bg-[var(--surface)] px-3 py-2 text-[13px] text-[var(--ink)] focus:outline-none focus:border-[var(--accent)]"
          >
            <option value="">Select a method…</option>
            {availableModels.map((m) => (
              <option key={m.model} value={m.model}>
                {m.model}
              </option>
            ))}
          </select>
        </div>

        {selectedEntry ? (
          <div className="space-y-4">
            {benchmarkRows.map((r) => (
              <div key={r.label}>
                <div className="flex items-baseline justify-between mb-1.5">
                  <span className="text-[13px] text-[var(--ink-2)]">
                    {r.label}
                  </span>
                  <span className="text-[13px] tnum">
                    <span className="font-semibold text-[var(--ink)]">
                      {r.value !== null ? r.value.toFixed(4) : "—"}
                    </span>
                    <span className="text-[var(--ink-3)]">
                      {" "}
                      / best {r.best.toFixed(4)}
                    </span>
                  </span>
                </div>
                <div
                  className="relative h-2.5 rounded-full overflow-visible"
                  style={{ background: "var(--bar-track)" }}
                >
                  {r.value !== null && (
                    <div
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{
                        width: `${Math.min(r.value * 100, 100)}%`,
                        background: "var(--series-ot)",
                      }}
                    />
                  )}
                  <div
                    className="absolute -top-1 -bottom-1 w-[2px] rounded-full"
                    style={{
                      left: `calc(${Math.min(r.best * 100, 100)}% - 1px)`,
                      background: "var(--ink-3)",
                    }}
                    title={`Best: ${r.best.toFixed(4)}`}
                  />
                </div>
              </div>
            ))}
            <p className="pt-2 text-[12px] text-[var(--ink-3)]">
              Bars and ticks use a 0–1 scale (F1 / recall).
            </p>
          </div>
        ) : (
          <EmptyState text="Select a method to view its profile." />
        )}
      </div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="h-64 flex items-center justify-center text-[14px] text-[var(--ink-3)]">
      {text}
    </div>
  );
}
