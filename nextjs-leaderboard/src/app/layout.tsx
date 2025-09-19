import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// Get the base URL for absolute URLs in metadata
const baseUrl = process.env.NEXT_PUBLIC_BASE_URL;

export const metadata: Metadata = {
  title: {
    default: "OTalign Leaderboard",
    template: "%s | OTalign",
  },
  description:
    "Interactive leaderboard for optimal transport-based protein sequence alignment algorithms. Compare performance across multiple datasets and metrics.",
  keywords: [
    "protein alignment",
    "optimal transport",
    "bioinformatics",
    "sequence alignment",
    "leaderboard",
    "benchmarking",
    "OTalign",
  ],
  authors: [{ name: "DeepFold" }],
  creator: "DeepFold",
  publisher: "OTalign",

  // Open Graph
  openGraph: {
    type: "website",
    locale: "en_US",
    siteName: "OTalign Leaderboard",
    title: "OTalign Leaderboard",
    description:
      "Interactive leaderboard for optimal transport-based protein sequence alignment algorithms",
    url: baseUrl,
    images: [
      {
        url: `${baseUrl}/og/og.png`,
        width: 1200,
        height: 600,
        alt: "OTalign Logo",
        type: "image/png",
      },
    ],
  },

  // Twitter Card
  twitter: {
    card: "summary_large_image",
    title: "OTalign Leaderboard",
    description:
      "Interactive leaderboard for optimal transport-based protein sequence alignment algorithms",
    images: [`${baseUrl}/svgs/og.png`],
    creator: "@otalign",
  },

  // Icons and manifest
  icons: {
    icon: [
      { url: "/web/favicon.ico", sizes: "any" },
      { url: "/web/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/web/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [
      { url: "/web/apple-touch-icon.png", sizes: "180x180", type: "image/png" },
    ],
    other: [
      { rel: "mask-icon", url: "/svgs/foreground.svg", color: "#000000" },
    ],
  },

  manifest: "/manifest.json",

  // Additional metadata
  category: "Science",
  classification: "Bioinformatics Tool",

  // Verification and SEO
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },

  // Viewport is handled by Next.js automatically in app directory
  other: {
    "theme-color": "#ffffff",
    "color-scheme": "light dark",
    "format-detection": "telephone=no",
  },
};

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* Preconnect to external domains for performance */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />

        {/* DNS prefetch for better performance */}
        <link rel="dns-prefetch" href="//vercel.app" />

        {/* Security headers */}
        <meta httpEquiv="X-Content-Type-Options" content="nosniff" />
        <meta httpEquiv="Referrer-Policy" content="origin-when-cross-origin" />

        {/* Additional meta tags for mobile */}
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="OTalign" />

        {/* Microsoft Tiles */}
        <meta name="msapplication-TileColor" content="#000000" />
        <meta name="msapplication-TileImage" content="/web/icon-512.png" />

        {/* Structured data for search engines */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebApplication",
              name: "OTalign Leaderboard",
              description:
                "Interactive leaderboard for optimal transport-based protein sequence alignment algorithms",
              url: "https://otalign.vercel.app",
              applicationCategory: "ScienceApplication",
              operatingSystem: "Any",
              offers: {
                "@type": "Offer",
                price: "0",
                priceCurrency: "USD",
              },
              creator: {
                "@type": "Organization",
                name: "DeepFold",
              },
            }),
          }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
