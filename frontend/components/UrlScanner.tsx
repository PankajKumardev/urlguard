"use client";

import { FormEvent, useState } from "react";
import { Search } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PredictionResponse, PredictionResult, ResultSkeleton } from "./PredictionResult";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:5000";

export function UrlScanner() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (loading) {
      return;
    }

    const trimmed = url.trim();
    if (!trimmed) {
      setError("Enter a URL to analyze.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: trimmed }),
        cache: "no-store",
      });
      const body = (await response.json().catch(() => null)) as { error?: string } | PredictionResponse | null;
      if (!response.ok) {
        throw new Error(body && "error" in body && body.error ? body.error : "The URL could not be analyzed.");
      }
      setResult(body as PredictionResponse);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "";
      if (message === "" || message === "Failed to fetch" || message.toLowerCase().includes("network")) {
        setError("The analysis service is unavailable. Start the API with python app.py, then try again.");
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Check a link</CardTitle>
          <CardDescription>HTTP and HTTPS only. The address is not opened.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-3">
            <div className="space-y-2">
              <Label htmlFor="url">URL</Label>
              <Input
                id="url"
                name="url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="Paste a URL to analyze..."
                autoComplete="off"
                autoCapitalize="off"
                spellCheck={false}
                className="h-11 md:text-base"
              />
            </div>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-muted-foreground" aria-live="polite">
                {loading ? "Analyzing URL..." : "Press Enter to analyze."}
              </p>
              <Button type="submit" size="lg" className="h-11 px-5" disabled={loading || url.trim() === ""}>
                <Search />
                {loading ? "Analyzing URL..." : "Analyze URL"}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {error ? (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      {loading ? <ResultSkeleton /> : null}
      {result ? <PredictionResult result={result} /> : null}
    </div>
  );
}
