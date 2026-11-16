import { ScannerHero } from "../components/ScannerHero";
import { Card, CardContent } from "@/components/ui/card";

export default function HomePage() {
  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 lg:py-10">
      <ScannerHero />
      <Card className="mt-8">
        <CardContent className="text-sm leading-relaxed text-muted-foreground">
          <p className="text-base font-medium text-foreground">After you submit</p>
          <p className="mt-2 max-w-[68ch]">
            The check reads the address, scores it with the saved model, and shows the class, the phishing score, and the traits that were measured. The page itself is never opened.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
