"use client"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"

export function MockBanner() {
  return (
    <Alert className="bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800">
      <AlertCircle className="w-4 h-4 text-blue-600 dark:text-blue-400" />
      <AlertDescription className="text-blue-800 dark:text-blue-200">
        This dashboard is using mock data for demonstration. Real-time data will appear once you connect your Alpaca
        account.
      </AlertDescription>
    </Alert>
  )
}
