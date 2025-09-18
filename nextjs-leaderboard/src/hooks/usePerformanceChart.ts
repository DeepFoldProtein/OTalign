import { useState, useEffect, useMemo } from "react";
import { LeaderboardEntry } from "@/lib/types";

interface ScatterDataPoint {
  x: number;
  y: number;
  model: string;
  type: string;
  organization: string;
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

  // Filter data with valid scores for scatter plot
  const validData = useMemo(
    () =>
      data.filter(
        (entry) => entry.malidup_f1 !== null && entry.malisam_f1 !== null
      ),
    [data]
  );

  const scatterData: ScatterDataPoint[] = useMemo(
    () =>
      validData.map((entry) => ({
        x: entry.malidup_f1!,
        y: entry.malisam_f1!,
        model: entry.model,
        type: entry.type,
        organization: entry.organization,
      })),
    [validData]
  );

  // Get color for different types
  const getTypeColor = (type: string) => {
    switch (type) {
      case "Traditional":
        return "#3B82F6"; // blue
      case "OTalign":
        return "#A855F7"; // purple
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
    () =>
      data.filter(
        (entry) => entry.malidup_f1 !== null || entry.malisam_f1 !== null
      ),
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
