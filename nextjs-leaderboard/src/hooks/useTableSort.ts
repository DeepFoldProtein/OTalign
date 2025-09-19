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
