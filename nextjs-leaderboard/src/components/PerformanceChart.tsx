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
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ReferenceLine,
  LabelList,
} from "recharts";
import { usePerformanceChart } from "@/hooks/usePerformanceChart";

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
  const {
    selectedModel,
    scatterData,
    availableModels,
    getRadarData,
    getTypeColor,
    setSelectedModel,
  } = usePerformanceChart({ data });

  // Separate data with valid parameters and N/A parameters
  const validParameterData = scatterData.filter((d) => d.x >= 0);
  const naParameterData = scatterData.filter((d) => d.x === -1);

  // Format parameter count for display
  const formatParameterCount = (value: number) => {
    if (value === -1) return "N/A";
    if (value >= 1000000000) return `${(value / 1000000000).toFixed(1)}B`;
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    return value.toString();
  };

  // Get specific colors for N/A parameter models to avoid overlapping
  const getNALineColor = (modelName: string) => {
    if (modelName.includes("Needleman-Wunsch")) {
      return "#EF4444"; // red
    } else if (modelName.includes("HHAlign")) {
      return "#F97316"; // orange
    } else {
      return "#6B7280"; // gray for other N/A models
    }
  };

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: Array<{ payload: ScatterDataPoint }>;
  }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-600 rounded-lg shadow-lg transition-none">
          <p className="font-semibold text-gray-900 dark:text-white">
            {data.model}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {data.organization}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Type: {data.type}
          </p>
          <p className="text-sm">Parameters: {data.parameters}</p>
          <p className="text-sm">Average Score: {data.y.toFixed(4)}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-8">
      {/* Scatter Plot */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 w-[800px]">
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
          Parameter Count vs Average Performance
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Compare model parameter counts against average performance scores.
        </p>

        {scatterData.length > 0 ? (
          <>
            <div className="h-[500px] w-full max-w-4xl mx-auto">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 60, right: 100, bottom: 60, left: 80 }}
                  data={[...validParameterData, ...naParameterData]}
                >
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="Parameters"
                    domain={
                      validParameterData.length > 0
                        ? [1000000, 15000000000]
                        : [0, 1]
                    }
                    scale="log"
                    tickFormatter={formatParameterCount}
                    ticks={Array.from(
                      new Set(validParameterData.map((d) => d.x))
                    ).sort((a, b) => a - b)}
                    label={{
                      value: "Parameter Count",
                      position: "insideBottom",
                      offset: -25,
                    }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="Average Score"
                    domain={[0, 1]}
                    tickFormatter={(value) => value.toFixed(2)}
                    label={{
                      value: "Average Score",
                      angle: -90,
                      position: "insideLeft",
                      style: { textAnchor: "middle" },
                    }}
                  />
                  <Tooltip content={<CustomTooltip />} animationDuration={0} />
                  <Legend verticalAlign="top" height={36} />

                  {/* Reference lines for N/A parameter models */}
                  {naParameterData.map((item, index) => (
                    <ReferenceLine
                      key={`na-${index}`}
                      y={item.y}
                      stroke={getNALineColor(item.model)}
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      label={false}
                    />
                  ))}

                  {/* Scatter points for models with valid parameter counts */}
                  {Array.from(
                    new Set(validParameterData.map((d) => d.type))
                  ).map((type) => (
                    <Scatter
                      key={type}
                      name={type}
                      data={validParameterData.filter((d) => d.type === type)}
                      fill={getTypeColor(type)}
                      strokeWidth={2}
                      stroke={getTypeColor(type)}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Custom legend for N/A parameter models */}
            {naParameterData.length > 0 && (
              <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                  Reference Methods without Protein Language Models
                </h4>
                <div className="flex flex-wrap gap-4">
                  {naParameterData.map((item, index) => (
                    <div
                      key={`legend-${index}`}
                      className="flex items-center gap-2"
                    >
                      <div
                        className="w-6 h-0.5 border-dashed border-2"
                        style={{ borderColor: getNALineColor(item.model) }}
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        {item.model} (Score: {item.y.toFixed(3)})
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="h-96 flex items-center justify-center text-gray-500 dark:text-gray-400">
            <div className="text-center">
              <div className="w-16 h-16 bg-gray-200 dark:bg-gray-600 rounded-full flex items-center justify-center mb-4 mx-auto">
                <svg
                  className="w-8 h-8"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <p>No performance data available yet</p>
            </div>
          </div>
        )}
      </div>

      {/* Radar Chart */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
          Model Performance Profile
        </h3>

        <div className="mb-4">
          <label
            htmlFor="model-select"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
          >
            Select Model:
          </label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
          >
            <option value="">Select a model...</option>
            {availableModels.map((model) => (
              <option key={model.model} value={model.model}>
                {model.model}
              </option>
            ))}
          </select>
        </div>

        {selectedModel ? (
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={getRadarData(selectedModel)}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 1]}
                  tickFormatter={(value) => value.toFixed(2)}
                />
                <Radar
                  name={selectedModel}
                  dataKey="value"
                  stroke="#3B82F6"
                  fill="#3B82F6"
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(4), "Score"]}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-96 flex items-center justify-center text-gray-500 dark:text-gray-400">
            <div className="text-center">
              <div className="w-16 h-16 bg-gray-200 dark:bg-gray-600 rounded-full flex items-center justify-center mb-4 mx-auto">
                <svg
                  className="w-8 h-8"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <p>Select a model to view its performance profile</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
