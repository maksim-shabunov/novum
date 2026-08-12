import type { Metadata } from "next";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

export const metadata: Metadata = {
  title: "NOVUM — mission control",
  description:
    "Onboard science-data triage for a planetary rover: what the rover captured, " +
    "what reached Earth, and the decision in between.",
};

/**
 * Dark by default and without a theme toggle: this is a ground-station console,
 * and an operator display that can be switched to white is a marketing page.
 *
 * No webfont. `next/font/google` self-hosts at runtime but still needs the
 * network at BUILD time, and the acceptance test is a clean machine running
 * `docker compose up`. System stacks cost nothing to load and never fail.
 */
export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full bg-background text-foreground">
        <TooltipProvider delay={120}>{children}</TooltipProvider>
      </body>
    </html>
  );
}
