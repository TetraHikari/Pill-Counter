import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PillTally",
  description: "Count pills from a phone photo",
};

export const viewport: Viewport = {
  themeColor: "#000000",
  viewportFit: "cover", // lets the page use the full screen under the notch (see safe-area padding)
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      {/* Browser extensions add attributes to <body> before React loads; ignore those (this element only). */}
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
