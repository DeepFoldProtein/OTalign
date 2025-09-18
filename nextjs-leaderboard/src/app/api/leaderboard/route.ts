import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export interface LeaderboardEntry {
  rank: number;
  model: string;
  type: string;
  description: string;
  paper_url?: string;
  code_url?: string;
  average_f1: number | null;
  malidup_f1: number | null;
  malisam_f1: number | null;
  sabmark_sup_recall: number | null;
  sabmark_twi_recall: number | null;
  malidup_recall: number | null;
  malisam_recall: number | null;
  date_submitted: string;
  organization: string;
}

export interface LeaderboardData {
  leaderboard_data: LeaderboardEntry[];
  metadata: {
    last_updated: string;
    total_models: number;
    datasets: string[];
    metrics: string[];
    version: string;
  };
}

export async function GET() {
  try {
    const dataPath = path.join(process.cwd(), 'public', 'data', 'benchmark_results.json');
    const fileContents = fs.readFileSync(dataPath, 'utf8');
    const data: LeaderboardData = JSON.parse(fileContents);
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error loading leaderboard data:', error);
    return NextResponse.json(
      { error: 'Failed to load leaderboard data' },
      { status: 500 }
    );
  }
}
