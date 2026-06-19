import type {
  HealthStatus,
  MarketSnapshot,
  ReportPayload,
  SessionState,
} from "@/lib/types";

export const ADAPTER_BASE_URL =
  process.env.NEXT_PUBLIC_ALPHASIGN_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export const API_BASE_URL = "/api/alphasign";

type RequestOptions = RequestInit & {
  signal?: AbortSignal;
};

export class AlphaSignApiError extends Error {
  status?: number;
  code?: string;

  constructor(message: string, status?: number, code?: string) {
    super(message);
    this.name = "AlphaSignApiError";
    this.status = status;
    this.code = code;
  }
}

export async function getHealth(signal?: AbortSignal): Promise<HealthStatus> {
  return request<HealthStatus>("/health", { signal });
}

export async function startSession(
  ticker: string,
  signal?: AbortSignal,
): Promise<SessionState> {
  return request<SessionState>("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker }),
    signal,
  });
}

export async function getSession(
  sessionId: string,
  signal?: AbortSignal,
): Promise<SessionState> {
  return request<SessionState>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    signal,
  });
}

export async function getMarketSnapshot(
  ticker: string,
  signal?: AbortSignal,
): Promise<MarketSnapshot> {
  return request<MarketSnapshot>(`/api/market/${encodeURIComponent(ticker)}`, {
    signal,
  });
}

export async function getReport(
  sessionId: string,
  signal?: AbortSignal,
): Promise<ReportPayload> {
  return request<ReportPayload>(
    `/api/sessions/${encodeURIComponent(sessionId)}/report`,
    { signal },
  );
}

export async function searchTickers(
  query: string,
  signal?: AbortSignal,
): Promise<string[]> {
  const result = await request<{ tickers?: string[] } | string[]>(
    `/api/tickers/search?q=${encodeURIComponent(query)}`,
    { signal },
  );
  return Array.isArray(result) ? result : result.tickers ?? [];
}

export function eventsUrl(sessionId: string) {
  return `${ADAPTER_BASE_URL}/api/sessions/${encodeURIComponent(sessionId)}/events`;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...options.headers,
      },
    });
  } catch (error) {
    throw new AlphaSignApiError(
      "Backend adapter is unreachable. Start the adapter or set NEXT_PUBLIC_ALPHASIGN_API_URL.",
      undefined,
      error instanceof Error ? error.name : "network_error",
    );
  }

  if (!response.ok) {
    const detail = await readError(response);
    throw new AlphaSignApiError(
      detail || `Adapter request failed with HTTP ${response.status}.`,
      response.status,
    );
  }

  const payload = (await response.json()) as T & {
    adapter_unavailable?: boolean;
    message?: string;
  };
  if (payload.adapter_unavailable) {
    throw new AlphaSignApiError(
      payload.message ??
        "Backend adapter is unreachable. Start the adapter or set ALPHASIGN_API_URL.",
      503,
      "adapter_unavailable",
    );
  }
  return payload;
}

async function readError(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string; message?: string };
    return payload.detail ?? payload.message ?? "";
  } catch {
    return "";
  }
}
