export interface LeaderboardEntry {
  rank: number;
  model: string;
  type: string;
  description: string;
  paper_url?: string;
  code_url?: string;
  average: number | null;
  malidup_f1: number | null;
  malisam_f1: number | null;
  sabmark_sup_recall: number | null;
  sabmark_twi_recall: number | null;
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

export interface SubmissionData {
  model: string;
  type: string;
  description: string;
  paper_url?: string;
  code_url: string;
  malidup_f1?: number;
  malisam_f1?: number;
  sabmark_sup_recall?: number;
  sabmark_twi_recall?: number;
  organization: string;
  date_submitted: string;
}
