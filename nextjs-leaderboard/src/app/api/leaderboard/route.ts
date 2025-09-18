import { NextResponse } from "next/server";
import { leaderboardData } from "@/lib/data";

export async function GET() {
  try {
    return NextResponse.json(leaderboardData);
  } catch (error) {
    console.error("Error loading leaderboard data:", error);
    return NextResponse.json(
      { error: "Failed to load leaderboard data" },
      { status: 500 }
    );
  }
}
