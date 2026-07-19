// Single source of truth for how each method family is colored + labeled.
// Colors reference the validated categorical palette in globals.css, so they
// resolve correctly in both light and dark mode.

export type MethodFamily = "OTalign" | "PLM-based" | "Traditional" | "Other";

export function methodColorVar(type: string): string {
  switch (type) {
    case "OTalign":
      return "var(--series-ot)";
    case "PLM-Based":
    case "PLM-based":
      return "var(--series-plm)";
    case "Traditional":
      return "var(--series-trad)";
    default:
      return "var(--ink-3)";
  }
}
