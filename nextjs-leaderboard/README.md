# OTalign Leaderboard

A Next.js-based leaderboard for evaluating protein structure alignment methods on standardized benchmark datasets.

## Features

- Interactive leaderboard with sorting and filtering
- Performance metrics visualization 
- Model submission interface
- Support for multiple evaluation datasets (MaliDup, MaliSAM, SABmark)
- Responsive design with modern UI components

## Getting Started

Install dependencies and run the development server:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to view the leaderboard.

## Deployment

This project is configured for deployment on Cloudflare Pages:

```bash
npm run preview  # Preview locally
npm run deploy   # Deploy to Cloudflare
```

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS
- **UI Components**: Headless UI, Heroicons
- **Charts**: Recharts
- **Deployment**: Cloudflare Pages via OpenNext
