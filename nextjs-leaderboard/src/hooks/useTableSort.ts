import { useState, useMemo } from "react";
import { LeaderboardEntry } from "@/lib/types";

export type SortField =
  | "rank"
  | "model"
  | "type"
  | "parameters"
  | "average"
  | "malidup_f1"
  | "malisam_f1"
  | "sabmark_sup_recall"
  | "sabmark_twi_recall"
  | "date_submitted";

export type SortDirection = "asc" | "desc";

interface UseTableSortProps {
  data: LeaderboardEntry[];
}

interface UseTableSortReturn {
  sortField: SortField;
  sortDirection: SortDirection;
  typeFilter: string;
  sortedAndFilteredData: LeaderboardEntry[];
  uniqueTypes: string[];
  handleSort: (field: SortField) => void;
  setTypeFilter: (filter: string) => void;
}

export function useTableSort({ data }: UseTableSortProps): UseTableSortReturn {
  const [sortField, setSortField] = useState<SortField>("rank");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");
  const [typeFilter, setTypeFilter] = useState<string>("all");

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

  const sortedAndFilteredData = useMemo(() => {
    let filtered = data;

    if (typeFilter !== "all") {
      filtered = data.filter((entry) => entry.type === typeFilter);
    }

    return filtered.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];

      // Handle null values
      if (aVal === null && bVal === null) return 0;
      if (aVal === null) return 1;
      if (bVal === null) return -1;

      // Special handling for parameters field
      if (sortField === "parameters") {
        const aParamCount = parseParameterCount(aVal as string);
        const bParamCount = parseParameterCount(bVal as string);

        // Handle N/A values (returned as -1)
        if (aParamCount === -1 && bParamCount === -1) return 0;
        if (aParamCount === -1) return 1;
        if (bParamCount === -1) return -1;

        return sortDirection === "asc"
          ? aParamCount - bParamCount
          : bParamCount - aParamCount;
      }

      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDirection === "asc"
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDirection === "asc" ? aVal - bVal : bVal - aVal;
      }

      return 0;
    });
  }, [data, sortField, sortDirection, typeFilter]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("asc");
    }
  };

  const uniqueTypes = useMemo(
    () => Array.from(new Set(data.map((entry) => entry.type))),
    [data]
  );

  return {
    sortField,
    sortDirection,
    typeFilter,
    sortedAndFilteredData,
    uniqueTypes,
    handleSort,
    setTypeFilter,
  };
}
