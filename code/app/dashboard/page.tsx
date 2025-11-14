import { DashboardNavbar } from "@/components/dashboard-navbar"
import { DashboardContent } from "@/components/dashboard-content"
import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Dashboard | Agentic E-Trading",
  description: "Your intelligent trading dashboard with real-time market insights",
}

export default function DashboardPage() {
  return (
    <>
      <DashboardNavbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <DashboardContent />
      </main>
    </>
  )
}
