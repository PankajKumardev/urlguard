"use client";

import { ShieldAlert, ShieldCheck } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress, ProgressLabel, ProgressValue } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";

export type PredictionResponse = {
  prediction: "SAFE" | "PHISHING";
  phishing_probability: number;
  model: string;
  reasons: string[];
  details: { label: string; value: string }[];
  analyzed_url: string;
  scheme_note: string | null;
  disclaimer: string;
};

export function PredictionResult({ result }: { result: PredictionResponse }) {
  const reduce = useReducedMotion();
  const phishing = result.prediction === "PHISHING";
  const percent = Math.max(0, Math.min(100, result.phishing_probability * 100));

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <Card>
        <CardHeader>
          <Badge variant={phishing ? "destructive" : "secondary"} className="w-fit">
            {phishing ? <ShieldAlert /> : <ShieldCheck />}
            {result.prediction}
          </Badge>
          <CardTitle className="font-heading text-2xl tracking-tight sm:text-3xl">
            {phishing ? "Potential Phishing Detected" : "Predicted Legitimate"}
          </CardTitle>
          <CardDescription className="max-w-[65ch] text-sm leading-relaxed">
            {phishing
              ? "The model identified URL patterns associated with phishing attempts."
              : "The model found a lower estimated likelihood of phishing. This does not guarantee that the website is safe."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <Progress
            value={percent}
            className={phishing ? "**:data-[slot=progress-indicator]:bg-destructive" : "**:data-[slot=progress-indicator]:bg-safe"}
          >
            <ProgressLabel>Estimated phishing probability</ProgressLabel>
            <ProgressValue>{() => `${percent.toFixed(1)}%`}</ProgressValue>
          </Progress>
          <p className="text-sm text-muted-foreground">
            Model score <span className="font-mono text-foreground">{result.phishing_probability.toFixed(4)}</span>
            <span className="mx-2 text-border">/</span>
            {result.model}
          </p>
          <Separator />
          <div>
            <p className="text-sm font-medium">Submitted URL</p>
            <p className="mt-2 break-all rounded-lg border bg-muted/40 px-3 py-2 font-mono text-sm">{result.analyzed_url}</p>
            {result.scheme_note ? <p className="mt-2 text-sm text-muted-foreground">{result.scheme_note}</p> : null}
          </div>
          <div>
            <p className="text-sm font-medium">URL characteristics</p>
            <dl className="mt-3 grid gap-3 sm:grid-cols-2">
              {result.details.map((item) => (
                <div key={item.label} className="rounded-lg border px-3 py-2">
                  <dt className="text-xs text-muted-foreground">{item.label}</dt>
                  <dd className="mt-1 text-sm font-medium">{item.value}</dd>
                </div>
              ))}
            </dl>
          </div>
          <div>
            <p className="text-sm font-medium">What the extractor noticed</p>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              These are indicators found while reading the URL. They do not, by themselves, prove why the model decided.
            </p>
            <ul className="mt-3 space-y-2">
              {result.reasons.map((reason) => (
                <li key={reason} className="text-sm leading-6">
                  {reason}
                </li>
              ))}
            </ul>
          </div>
          <Alert>
            <AlertDescription>{result.disclaimer}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function ResultSkeleton() {
  return (
    <Card aria-hidden="true">
      <CardHeader>
        <Skeleton className="h-5 w-16" />
        <Skeleton className="h-8 w-64 max-w-full" />
        <Skeleton className="h-4 w-full max-w-md" />
      </CardHeader>
      <CardContent className="space-y-3">
        <Skeleton className="h-2 w-full" />
        <Skeleton className="h-16 w-full" />
      </CardContent>
    </Card>
  );
}
