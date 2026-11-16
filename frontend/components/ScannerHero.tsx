"use client";

import { motion, useReducedMotion } from "motion/react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { UrlScanner } from "./UrlScanner";

const signals = [
  "URL length",
  "Dots",
  "Hyphens",
  "Digits",
  "@ symbols",
  "Subdomains",
  "HTTPS",
  "IP address",
  "Keywords",
  "Question marks",
  "Equals signs",
  "Path length",
];

export function ScannerHero() {
  const reduce = useReducedMotion();

  return (
    <section className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(16rem,0.8fr)] lg:gap-8">
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="space-y-6"
      >
        <div>
          <h1 className="max-w-[12ch] text-5xl font-semibold tracking-tight text-balance sm:text-6xl">
            Know before you click.
          </h1>
          <p className="mt-4 max-w-[46ch] text-lg leading-relaxed text-muted-foreground">
            Check suspicious links using machine learning and understand the signals behind each prediction.
          </p>
        </div>
        <UrlScanner />
      </motion.div>

      <motion.div
        initial={reduce ? false : { opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, delay: reduce ? 0 : 0.08, ease: [0.16, 1, 0.3, 1] }}
        className="lg:sticky lg:top-24"
      >
        <Card>
          <CardHeader>
            <CardTitle>What the model sees</CardTitle>
            <CardDescription>
              Twelve numbers from the URL text. Training and this check use the same function.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="grid grid-cols-2 gap-2">
              {signals.map((name) => (
                <li key={name} className="rounded-lg bg-muted px-3 py-2 text-sm font-medium">
                  {name}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </motion.div>
    </section>
  );
}
