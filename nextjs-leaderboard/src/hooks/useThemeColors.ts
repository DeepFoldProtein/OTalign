import { useEffect, useState } from "react";

// Recharts writes colors as SVG *presentation attributes* (fill="…"), where
// CSS var() does not resolve. So we read the token values off :root at runtime
// and hand recharts concrete hex, re-reading when the color scheme flips.
const TOKENS = {
  seriesOt: "--series-ot",
  seriesPlm: "--series-plm",
  seriesTrad: "--series-trad",
  axis: "--ink-3",
  grid: "--line",
  surface: "--surface",
  accent: "--accent",
} as const;

export type ThemeColors = Record<keyof typeof TOKENS, string>;

const FALLBACK: ThemeColors = {
  seriesOt: "#0064ff",
  seriesPlm: "#008300",
  seriesTrad: "#7c5cff",
  axis: "#949aa6",
  grid: "#e7e9ee",
  surface: "#ffffff",
  accent: "#0064ff",
};

function read(): ThemeColors {
  if (typeof window === "undefined") return FALLBACK;
  const cs = getComputedStyle(document.documentElement);
  const out = {} as ThemeColors;
  (Object.keys(TOKENS) as (keyof typeof TOKENS)[]).forEach((k) => {
    out[k] = cs.getPropertyValue(TOKENS[k]).trim() || FALLBACK[k];
  });
  return out;
}

export function useThemeColors(): ThemeColors {
  const [colors, setColors] = useState<ThemeColors>(FALLBACK);

  useEffect(() => {
    setColors(read());
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setColors(read());
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  return colors;
}
