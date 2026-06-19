const COOKIE_NAME = "alphasign.adapter_url";
const DEFAULT_URL =
  process.env.NEXT_PUBLIC_ALPHASIGN_API_URL?.replace(/\/$/, "") ?? "http://localhost:8765";

function normalizeUrl(value: string) {
  const normalized = value.trim().replace(/\/$/, "");
  if (
    typeof window !== "undefined" &&
    window.location.protocol === "https:" &&
    normalized.startsWith("http://")
  ) {
    const hostname = new URL(normalized).hostname;
    if (hostname !== "localhost" && hostname !== "127.0.0.1" && hostname !== "::1") {
      return normalized.replace(/^http:\/\//, "https://");
    }
  }
  return normalized;
}

export function getAdapterUrl() {
  if (typeof window === "undefined") return DEFAULT_URL;
  const cookie = document.cookie
    .split("; ")
    .find((entry) => entry.startsWith(`${COOKIE_NAME}=`));
  const stored = cookie ? decodeURIComponent(cookie.split("=").slice(1).join("=")) : "";
  return normalizeUrl(stored) || DEFAULT_URL;
}

export function setAdapterUrl(url: string) {
  if (typeof window === "undefined") return;
  const value = normalizeUrl(url);
  if (!value) return;
  document.cookie = `${COOKIE_NAME}=${encodeURIComponent(value)}; path=/; max-age=31536000; samesite=lax`;
}

export function clearAdapterUrl() {
  if (typeof window === "undefined") return;
  document.cookie = `${COOKIE_NAME}=; path=/; max-age=0; samesite=lax`;
}

export function readAdapterUrlCookie(cookieHeader: string | null | undefined) {
  if (!cookieHeader) return "";
  for (const entry of cookieHeader.split(";")) {
    const [rawKey, ...rest] = entry.trim().split("=");
    if (rawKey === COOKIE_NAME) {
      return normalizeUrl(decodeURIComponent(rest.join("=")));
    }
  }
  return "";
}
