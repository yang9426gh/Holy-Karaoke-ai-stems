import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Clawd Karaoke AI",
  description: "Apple Music style karaoke with AI voice separation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
