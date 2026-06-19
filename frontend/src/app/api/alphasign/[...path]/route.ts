import { NextRequest } from "next/server";

const adapterBaseUrl =
  process.env.ALPHASIGN_API_URL?.replace(/\/$/, "") ??
  process.env.NEXT_PUBLIC_ALPHASIGN_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

export async function GET(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

async function proxyRequest(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  const upstreamUrl = new URL(`${adapterBaseUrl}/${path.join("/")}`);
  upstreamUrl.search = request.nextUrl.search;

  try {
    const upstream = await fetch(upstreamUrl, {
      method: request.method,
      headers: copyHeaders(request),
      body: request.method === "GET" ? undefined : await request.text(),
      cache: "no-store",
      signal: request.signal,
    });

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: copyResponseHeaders(upstream),
    });
  } catch {
    return Response.json(
      {
        status: "unavailable",
        adapter_unavailable: true,
        message:
          "Backend adapter is unreachable. Start the adapter or set ALPHASIGN_API_URL.",
      },
      { status: 200 },
    );
  }
}

function copyHeaders(request: NextRequest) {
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  headers.set("accept", request.headers.get("accept") ?? "application/json");
  return headers;
}

function copyResponseHeaders(response: Response) {
  const headers = new Headers();
  const contentType = response.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const cacheControl = response.headers.get("cache-control");
  headers.set("cache-control", cacheControl ?? "no-store");
  return headers;
}
