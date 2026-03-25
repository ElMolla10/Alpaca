"use client"

import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { KPICard } from "@/components/kpi-card"
import { EquityChart } from "@/components/equity-chart"
import { PositionsTable } from "@/components/positions-table"
import { TradesTable } from "@/components/trades-table"
interface Summary {
  equity: number
  cash: number
  buying_power: number
  total_pnl: number
  total_pnl_pct: number
  max_drawdown: number
  win_rate: number
  sharpe: number
  trades_today: number
}

export function DashboardContent() {
  const [activeTab, setActiveTab] = useState("overview")
  const [summary, setSummary] = useState<Summary | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch("/api/dashboard/summary")
      .then((r) => r.json())
      .then((data) => { if (!data.error) setSummary(data) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="positions">Positions</TabsTrigger>
          <TabsTrigger value="trades">Trades</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            <KPICard title="Total Equity" value={summary?.equity ?? 0} unit="USD" format="currency" loading={loading || !summary} />
            <KPICard title="Available Cash" value={summary?.cash ?? 0} unit="USD" format="currency" loading={loading || !summary} />
            <KPICard title="Total P&L %" value={summary?.total_pnl_pct ?? 0} change={summary?.total_pnl_pct} format="percent" loading={loading || !summary} />
            <KPICard title="Max Drawdown" value={summary?.max_drawdown ?? 0} change={summary ? -summary.max_drawdown : undefined} format="percent" loading={loading || !summary} />
            <KPICard title="Win Rate" value={summary?.win_rate ?? 0} format="percent" loading={loading || !summary} />
            <KPICard title="Total P&L $" value={summary?.total_pnl ?? 0} format="currency" loading={loading || !summary} />
            <KPICard title="Buying Power" value={summary?.buying_power ?? 0} format="currency" loading={loading || !summary} />
            <KPICard title="Sharpe Ratio" value={summary?.sharpe ?? 0} format="number" loading={loading || !summary} />
            <KPICard title="Trades Today" value={summary?.trades_today ?? 0} format="number" loading={loading || !summary} />
          </div>

          <EquityChart />
        </TabsContent>

        <TabsContent value="positions">
          <PositionsTable />
        </TabsContent>

        <TabsContent value="trades">
          <TradesTable />
        </TabsContent>
      </Tabs>
    </div>
  )
}
