"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getHealth,
  getMarketSnapshot,
  getReport,
  getSession,
  startSession,
} from "@/lib/api";
import {
  fixtureEvents,
  fixtureMarket,
  fixtureReport,
  fixtureSession,
} from "@/lib/fixtures";
import { normalizeReport } from "@/lib/report-parser";
import type {
  AgentEvent,
  HealthStatus,
  MarketSnapshot,
  ReportPayload,
  SessionState,
} from "@/lib/types";

type LoadState = "idle" | "loading" | "ready" | "error";

export function useAnalysisSession() {
  const startInFlightRef = useRef(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthState, setHealthState] = useState<LoadState>("idle");
  const [session, setSession] = useState<SessionState | null>(null);
  const [market, setMarket] = useState<MarketSnapshot | null>(null);
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [fixtureMode, setFixtureMode] = useState(false);
  const [retryAfter, setRetryAfter] = useState<number | null>(null);
  const [isCoolingDown, setIsCoolingDown] = useState(false);

  const refreshHealth = useCallback(async () => {
    const controller = new AbortController();
    setHealthState("loading");
    try {
      const value = await getHealth(controller.signal);
      setHealth(value);
      setHealthState("ready");
      setRetryAfter(null);
      setIsCoolingDown(false);
      setError(null);
    } catch (healthError) {
      setHealth({
        status: "unavailable",
        band_configured: false,
        message:
          healthError instanceof Error
            ? healthError.message
            : "Backend adapter health check failed.",
      });
      setHealthState("error");
    }
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshHealth();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [refreshHealth]);

  useEffect(() => {
    if (retryAfter == null) return;
    const remaining = retryAfter - Date.now();
    if (remaining <= 0) {
      const timer = window.setTimeout(() => {
        setIsCoolingDown(false);
        setRetryAfter(null);
      }, 0);
      return () => window.clearTimeout(timer);
    }
    const timer = window.setTimeout(() => {
      setIsCoolingDown(false);
      setRetryAfter(null);
    }, remaining);
    return () => window.clearTimeout(timer);
  }, [retryAfter]);

  const beginSession = useCallback(async (ticker: string) => {
    if (startInFlightRef.current) return;

    const retryBlocked = retryAfter != null && Date.now() < retryAfter;
    if (retryBlocked) {
      setError("Session start is cooling down after a failed adapter request.");
      return;
    }

    if (health?.status !== "ok") {
      setError("Connect the backend adapter before starting a live Band session.");
      return;
    }

    const normalizedTicker = ticker.trim().toUpperCase();
    if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(normalizedTicker)) {
      setError("Enter a valid ticker symbol.");
      return;
    }

    startInFlightRef.current = true;
    setIsStarting(true);
    setFixtureMode(false);
    setError(null);
    setReport(null);
    setEvents([]);

    const controller = new AbortController();
    try {
      const nextSession = await startSession(normalizedTicker, controller.signal);
      setSession(nextSession);
      const [nextMarket, nextReport] = await Promise.allSettled([
        getMarketSnapshot(normalizedTicker, controller.signal),
        nextSession.report_ready
          ? getReport(nextSession.session_id, controller.signal)
          : Promise.resolve(null),
      ]);

      if (nextMarket.status === "fulfilled") setMarket(nextMarket.value);
      if (nextReport.status === "fulfilled" && nextReport.value) {
        setReport(normalizeReport(nextReport.value));
      }
    } catch (sessionError) {
      setError(
        sessionError instanceof Error
          ? sessionError.message
          : "Could not start analysis session.",
      );
      setIsCoolingDown(true);
      setRetryAfter(Date.now() + 3000);
    } finally {
      startInFlightRef.current = false;
      setIsStarting(false);
    }
  }, [health?.status, retryAfter]);

  const loadFixturePreview = useCallback(() => {
    setFixtureMode(true);
    setError(null);
    setHealth({
      status: "degraded",
      band_configured: false,
      message: "Using isolated frontend fixtures. Backend adapter is not connected.",
    });
    setSession(fixtureSession);
    setMarket(fixtureMarket);
    setEvents(fixtureEvents);
    setReport(normalizeReport(fixtureReport));
  }, []);

  const appendEvent = useCallback((event: AgentEvent) => {
    setEvents((current) => {
      if (current.some((item) => item.id === event.id)) return current;
      return [...current, event].slice(-80);
    });
  }, []);

  useEffect(() => {
    if (!session || fixtureMode) return;
    if (session.status === "completed" || session.status === "failed") return;

    const interval = window.setInterval(async () => {
      try {
        const nextSession = await getSession(session.session_id);
        setSession(nextSession);
        if (nextSession.report_ready) {
          const nextReport = await getReport(nextSession.session_id);
          setReport(normalizeReport(nextReport));
        }
      } catch (pollError) {
        setError(
          pollError instanceof Error
            ? pollError.message
            : "Polling session state failed.",
        );
      }
    }, 4000);

    return () => window.clearInterval(interval);
  }, [fixtureMode, session]);

  const connectionLabel = useMemo(() => {
    if (healthState === "loading") return "Checking";
    if (health?.status === "ok") return "Connected";
    if (health?.status === "degraded") return "Degraded";
    return "Disconnected";
  }, [health, healthState]);

  const canStartSession = useMemo(() => {
    if (isStarting) return false;
    if (health?.status !== "ok") return false;
    if (isCoolingDown) return false;
    return true;
  }, [health?.status, isCoolingDown, isStarting]);

  const startDisabledReason = useMemo(() => {
    if (isStarting) return "Starting session...";
    if (isCoolingDown) return "Cooling down after a failed adapter request.";
    if (healthState === "loading") return "Checking backend adapter...";
    if (health?.status !== "ok") return "Backend adapter is disconnected.";
    return null;
  }, [health?.status, healthState, isCoolingDown, isStarting]);

  return {
    appendEvent,
    beginSession,
    canStartSession,
    connectionLabel,
    error,
    events,
    fixtureMode,
    health,
    healthState,
    isStarting,
    loadFixturePreview,
    market,
    refreshHealth,
    report,
    session,
    startDisabledReason,
  };
}
