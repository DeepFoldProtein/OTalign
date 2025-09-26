import { useState, useEffect, useMemo } from "react";
import { LeaderboardEntry } from "@/lib/types";

interface ScatterDataPoint {
  x: number;
  y: number;
  model: string;
  type: string;
  organization: string;
  parameters: string;
}

interface RadarDataPoint {
  metric: string;
  value: number;
  fullMark: 1;
}

interface UsePerformanceChartProps {
  data: LeaderboardEntry[];
}

interface UsePerformanceChartReturn {
  selectedModel: string;
  scatterData: ScatterDataPoint[];
  availableModels: LeaderboardEntry[];
  getRadarData: (modelName: string) => RadarDataPoint[];
  getTypeColor: (type: string) => string;
  setSelectedModel: (model: string) => void;
}

export function usePerformanceChart({
  data,
}: UsePerformanceChartProps): UsePerformanceChartReturn {
  const [selectedModel, setSelectedModel] = useState<string>("");

  // Auto-select the first place model when data changes
  useEffect(() => {
    const firstPlaceModel = data.find((entry) => entry.rank === 1)?.model || "";
    if (firstPlaceModel && !selectedModel) {
      setSelectedModel(firstPlaceModel);
    }
  }, [data, selectedModel]);

  // Parse parameter count from string (e.g., "1.2B" -> 1200, "N/A" -> -1)
  const parseParameterCount = (paramStr: string): number => {
    if (paramStr === "N/A" || paramStr === "n/a" || !paramStr) {
      return -1; // Special value for N/A
    }

    const cleanStr = paramStr.toLowerCase().replace(/[,\s]/g, "");
    // Updated regex to handle cases like "100B INT4" by matching the number and unit at the beginning
    const match = cleanStr.match(/^(\d+(?:\.\d+)?)(k|m|b)/);

    if (!match) return -1;

    const value = parseFloat(match[1]);
    const unit = match[2];

    switch (unit) {
      case "k":
        return value * 1000;
      case "m":
        return value * 1000000;
      case "b":
        return value * 1000000000;
      default:
        return value;
    }
  };

  // Filter data with valid average scores for scatter plot
  const validData = useMemo(
    () => data.filter((entry) => entry.average !== null),
    [data]
  );

  const scatterData: ScatterDataPoint[] = useMemo(() => {
    const data = validData.map((entry) => ({
      x: parseParameterCount(entry.parameters),
      y: entry.average!,
      model: entry.model,
      type: entry.type,
      organization: entry.organization,
      parameters: entry.parameters,
    }));
    console.log("Scatter data:", data); // Debug log
    return data;
  }, [validData]);

  // Get color for different types
  const getTypeColor = (type: string) => {
    switch (type) {
      case "Traditional":
        return "#3B82F6"; // blue
      case "OTalign":
        return "#A855F7"; // purple
      case "PLM-Based":
        return "#10B981"; // green
      case "PLM-based":
        return "#10B981"; // green
      default:
        return "#6B7280"; // gray
    }
  };

  // Prepare radar chart data for selected model
  const getRadarData = (modelName: string): RadarDataPoint[] => {
    const model = data.find((entry) => entry.model === modelName);
    if (!model) return [];

    return [
      {
        metric: "MALIDUP F1",
        value: model.malidup_f1 || 0,
        fullMark: 1,
      },
      {
        metric: "MALISAM F1",
        value: model.malisam_f1 || 0,
        fullMark: 1,
      },
      {
        metric: "SABmark (sup)",
        value: model.sabmark_sup_recall || 0,
        fullMark: 1,
      },
      {
        metric: "SABmark (twi)",
        value: model.sabmark_twi_recall || 0,
        fullMark: 1,
      },
    ];
  };

  const availableModels = useMemo(
    () => data.filter((entry) => entry.average !== null),
    [data]
  );

  return {
    selectedModel,
    scatterData,
    availableModels,
    getRadarData,
    getTypeColor,
    setSelectedModel,
  };
}
