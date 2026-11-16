import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SiteHeader } from "../components/SiteHeader";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const sans = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
  display: "swap",
});

const mono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "URLGuard",
  description: "Educational machine-learning check for suspicious URLs.",
};

const themeScript = `(function(){try{var t=localStorage.getItem("urlguard-theme");if(t!=="light"&&t!=="dark"){t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}document.documentElement.dataset.theme=t;document.documentElement.classList.toggle("dark",t==="dark");}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${sans.variable} ${mono.variable}`}>
      <body className="min-h-[100dvh] bg-background font-sans text-foreground antialiased">
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <TooltipProvider>
          <div className="flex min-h-[100dvh] flex-col">
            <SiteHeader />
            <main className="flex-1">{children}</main>
            <footer className="border-t">
              <div className="mx-auto grid max-w-6xl gap-1 px-4 py-8 text-sm text-muted-foreground sm:px-6">
                <p className="text-foreground">Jaypee Institute of Information Technology, Noida</p>
                <p>Department of Computer Applications</p>
                <p>MCA, first year, Machine Learning Lab</p>
                <p>URLGuard is an educational estimate. It is not a security guarantee.</p>
              </div>
            </footer>
          </div>
        </TooltipProvider>
      </body>
    </html>
  );
}
