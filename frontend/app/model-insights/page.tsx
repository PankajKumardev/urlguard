"use client";

import { useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:5000";

type ScoreRow = {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
};

type ModelInfo = {
  model_name: string;
  generated_at: string;
  selection_reason: string;
  feature_names: string[];
  features: { name: string; label: string; description: string; why: string }[];
  dataset: {
    name: string;
    official_url: string;
    license: string;
    citation: string;
    published_rows: number;
    published_legitimate: number;
    published_phishing: number;
  };
  cleaning: {
    original_rows: number;
    final_rows: number;
    safe_count: number;
    phishing_count: number;
  };
  counts: {
    modeling_rows: number;
    modeling_safe: number;
    modeling_phishing: number;
    train_rows: number;
    validation_rows: number;
    test_rows: number;
  };
  validation_comparison: ScoreRow[];
  test_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
  };
  confusion_matrix_image: string;
  comparison_chart_image: string;
};

function four(value: number) {
  return value.toFixed(4);
}

function percent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function comma(value: number) {
  return value.toLocaleString("en-US");
}

export default function ModelInsightsPage() {
  const [data, setData] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await fetch(`${API}/api/model-info`, { cache: "no-store" });
        const body = (await response.json().catch(() => null)) as (ModelInfo & { error?: string }) | null;
        if (!response.ok) {
          throw new Error(body?.error || "Model information is unavailable.");
        }
        if (!cancelled && body) {
          setData(body);
        }
      } catch (caught) {
        if (cancelled) {
          return;
        }
        const message = caught instanceof Error ? caught.message : "";
        if (message === "" || message === "Failed to fetch" || message.toLowerCase().includes("network")) {
          setError("Model insights could not be loaded. Start the API with python app.py, then refresh this page.");
        } else {
          setError(message);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const scores = data
    ? [
        ["Accuracy", data.test_metrics.accuracy],
        ["Precision", data.test_metrics.precision],
        ["Recall", data.test_metrics.recall],
        ["F1-score", data.test_metrics.f1],
      ]
    : [];

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 lg:py-10">
      <h1 className="max-w-[12ch] text-5xl font-semibold tracking-tight">Model Insights</h1>
      <p className="mt-4 max-w-[62ch] text-lg leading-relaxed text-muted-foreground">
        Three models were compared on the validation split. The held-out test split was scored once, after the model was chosen.
      </p>

      {error ? (
        <Alert variant="destructive" className="mt-6">
          <AlertTitle>Could not load results</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      {!data && !error ? (
        <div className="mt-8 space-y-3" aria-hidden="true">
          <Skeleton className="h-10 w-72" />
          <Skeleton className="h-40 w-full" />
        </div>
      ) : null}

      {data ? (
        <Tabs defaultValue="overview" className="mt-8">
          <TabsList className="h-auto flex-wrap">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="comparison">Comparison</TabsTrigger>
            <TabsTrigger value="matrix">Matrix</TabsTrigger>
            <TabsTrigger value="data">Dataset</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <Alert>
              <AlertDescription>
                These scores come from a stratified sample of {comma(data.counts.modeling_rows)} real records, not from all{" "}
                {comma(data.dataset.published_rows)} published rows. Generated {data.generated_at}.
              </AlertDescription>
            </Alert>
            <Card>
              <CardHeader>
                <CardTitle>{data.model_name}</CardTitle>
                <CardDescription>Held-out test performance. Phishing is the positive class.</CardDescription>
              </CardHeader>
              <CardContent>
                <dl className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                  {scores.map(([label, value]) => (
                    <div key={String(label)}>
                      <dt className="text-xs text-muted-foreground">{label}</dt>
                      <dd className="mt-1 font-mono text-3xl tracking-tight">{percent(Number(value))}</dd>
                      <p className="font-mono text-xs text-muted-foreground">{four(Number(value))}</p>
                    </div>
                  ))}
                </dl>
                <Separator className="my-5" />
                <dl className="grid gap-3 sm:grid-cols-3">
                  {[
                    ["Features", String(data.feature_names.length)],
                    ["Training rows", comma(data.counts.train_rows)],
                    ["Validation rows", comma(data.counts.validation_rows)],
                    ["Test rows", comma(data.counts.test_rows)],
                    ["Dataset", data.dataset.name],
                    ["License", data.dataset.license],
                  ].map(([label, value]) => (
                    <div key={label}>
                      <dt className="text-xs text-muted-foreground">{label}</dt>
                      <dd className="mt-1 text-sm font-medium">{value}</dd>
                    </div>
                  ))}
                </dl>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="comparison" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Validation comparison</CardTitle>
                <CardDescription>{data.selection_reason}</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Model</TableHead>
                      <TableHead>Accuracy</TableHead>
                      <TableHead>Precision</TableHead>
                      <TableHead>Recall</TableHead>
                      <TableHead>F1</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.validation_comparison.map((row) => {
                      const selected = row.model === data.model_name;
                      return (
                        <TableRow key={row.model} data-state={selected ? "selected" : undefined}>
                          <TableCell className="font-medium">
                            {row.model}
                            {selected ? (
                              <Badge variant="secondary" className="ml-2">
                                Selected
                              </Badge>
                            ) : null}
                          </TableCell>
                          <TableCell className="font-mono">{four(row.accuracy)}</TableCell>
                          <TableCell className="font-mono">{four(row.precision)}</TableCell>
                          <TableCell className="font-mono">{four(row.recall)}</TableCell>
                          <TableCell className="font-mono">{four(row.f1)}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
            <Card className="overflow-hidden">
              <img
                src={`${API}${data.comparison_chart_image}?generated=${encodeURIComponent(data.generated_at)}`}
                alt="Bar chart of validation accuracy, precision, recall, and F1. The axis starts at zero."
                className="w-full"
              />
            </Card>
          </TabsContent>

          <TabsContent value="matrix">
            <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
              <Card className="overflow-hidden">
                <CardHeader>
                  <CardTitle>Confusion matrix</CardTitle>
                  <CardDescription>
                    Test-set counts for {data.model_name}. A false positive is a legitimate URL flagged as phishing. A false negative is a phishing URL called safe.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <img
                    src={`${API}${data.confusion_matrix_image}?generated=${encodeURIComponent(data.generated_at)}`}
                    alt="Confusion matrix for the selected model on the held-out test set."
                    className="w-full rounded-lg"
                  />
                </CardContent>
              </Card>
              <Card>
                <CardContent className="grid grid-cols-2 gap-4 pt-4">
                  {[
                    ["True positive", data.test_metrics.true_positive, "Phishing caught"],
                    ["False positive", data.test_metrics.false_positive, "Legitimate flagged"],
                    ["False negative", data.test_metrics.false_negative, "Phishing missed"],
                    ["True negative", data.test_metrics.true_negative, "Legitimate kept"],
                  ].map(([label, value, text]) => (
                    <div key={String(label)}>
                      <p className="text-xs text-muted-foreground">{label}</p>
                      <p className="mt-1 font-mono text-3xl">{comma(Number(value))}</p>
                      <p className="text-sm text-muted-foreground">{text}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="data" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Dataset</CardTitle>
                <CardDescription>
                  Published class sizes: {comma(data.dataset.published_legitimate)} legitimate and{" "}
                  {comma(data.dataset.published_phishing)} phishing.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <dl className="grid gap-3 sm:grid-cols-2">
                  {[
                    ["Name", data.dataset.name],
                    ["License", data.dataset.license],
                    ["Original records", comma(data.cleaning.original_rows)],
                    ["Records used", comma(data.counts.modeling_rows)],
                    ["SAFE in the sample", comma(data.counts.modeling_safe)],
                    ["PHISHING in the sample", comma(data.counts.modeling_phishing)],
                    ["Cleaned SAFE labels", comma(data.cleaning.safe_count)],
                    ["Cleaned PHISHING labels", comma(data.cleaning.phishing_count)],
                  ].map(([label, value]) => (
                    <div key={label} className="border-b pb-3">
                      <dt className="text-xs text-muted-foreground">{label}</dt>
                      <dd className="mt-1 text-sm">{value}</dd>
                    </div>
                  ))}
                </dl>
                <p className="mt-4 text-sm leading-6 text-muted-foreground">
                  Source:{" "}
                  <a className="text-primary underline underline-offset-4" href={data.dataset.official_url} target="_blank" rel="noreferrer">
                    UCI Machine Learning Repository
                  </a>
                  . {data.dataset.citation}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Feature extraction</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Feature</TableHead>
                      <TableHead>How it is calculated</TableHead>
                      <TableHead>Why it may help</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.features.map((feature) => (
                      <TableRow key={feature.name}>
                        <TableCell className="whitespace-normal font-medium">{feature.label}</TableCell>
                        <TableCell className="whitespace-normal text-muted-foreground">{feature.description}</TableCell>
                        <TableCell className="whitespace-normal text-muted-foreground">{feature.why}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      ) : null}
    </div>
  );
}
