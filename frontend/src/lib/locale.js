export const DEFAULT_LOCALE = "zh-CN";

export function normalizeLocaleHint(value) {
  const text = String(value || "").trim().replaceAll("_", "-");
  if (!text || text.length > 64) return "";
  try {
    return Intl.getCanonicalLocales(text)[0] || "";
  } catch {
    return "";
  }
}

export function getPreferredLocales(source = globalThis.navigator) {
  const values = Array.isArray(source?.languages) && source.languages.length
    ? source.languages
    : [source?.language];
  const locales = [];
  for (const value of values) {
    const locale = normalizeLocaleHint(value);
    if (locale && !locales.includes(locale)) locales.push(locale);
    if (locales.length >= 3) break;
  }
  return locales.length ? locales : [DEFAULT_LOCALE];
}

export function getLocaleHint(source = globalThis.navigator) {
  return getPreferredLocales(source)[0] || DEFAULT_LOCALE;
}
