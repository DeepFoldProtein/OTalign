import { useState, useEffect } from "react";
import { LeaderboardData } from "@/lib/types";

interface UseLeaderboardReturn {
  data: LeaderboardData | null;
  loading: boolean;
  error: string | null;
}

export function useLeaderboard(): UseLeaderboardReturn {
  const [data, setData] = useState<LeaderboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/api/leaderboard");
        if (!response.ok) {
          throw new Error("Failed to fetch leaderboard data");
        }
        const leaderboardData = await response.json();
        setData(leaderboardData);
        setError(null);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load leaderboard data"
        );
        setData(null);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return { data, loading, error };
}
