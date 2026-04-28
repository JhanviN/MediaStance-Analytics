export const ADV_COLOR = "#E24B4A";
export const COOP_COLOR = "#639922";
export const NEUT_COLOR = "#378ADD";

export const LABEL_COLORS: Record<string, string> = {
  adversarial: ADV_COLOR,
  cooperative: COOP_COLOR,
  neutral: NEUT_COLOR,
};

export const ALL_PAIRS = [
  "CN-IN", "IN-US", "CN-US", "IN-RU", "IN-PK",
  "IN-IR", "IL-IN", "CN-RU", "CN-IR", "CN-PK",
  "RU-US", "IR-US", "IL-US", "IL-IR", "IR-RU",
];

export const COUNTRY_NAMES: Record<string, string> = {
  CN: "China",
  IN: "India",
  US: "USA",
  RU: "Russia",
  PK: "Pakistan",
  IR: "Iran",
  IL: "Israel",
};

/** "CN-US" → "China – USA" */
export function pairLabel(pair: string): string {
  const [a, b] = pair.split("-");
  return `${COUNTRY_NAMES[a] ?? a} – ${COUNTRY_NAMES[b] ?? b}`;
}
