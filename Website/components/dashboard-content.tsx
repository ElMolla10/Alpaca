"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { KPICard } from "@/components/kpi-card"
import { EquityChart } from "@/components/equity-chart"
import { PositionsTable } from "@/components/positions-table"
import { TradesTable } from "@/components/trades-table"
import { mockPortfolioSummary } from "@/lib/mock-data"

export function DashboardContent() {
  const [activeTab, setActiveTab] = useState("overview")

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
            <KPICard title="Total Equity" value={mockPortfolioSummary.totalEquity} unit="USD" format="currency" />
            <KPICard title="Available Cash" value={mockPortfolioSummary.cash} unit="USD" format="currency" />
            <KPICard
              title="Day P&L %"
              value={mockPortfolioSummary.dayReturnPercentage}
              change={mockPortfolioSummary.dayReturnPercentage}
              format="percent"
            />
            <KPICard
              title="Total Return %"
              value={mockPortfolioSummary.returnPercentage}
              change={mockPortfolioSummary.returnPercentage}
              format="percent"
            />
            <KPICard
              title="Max Drawdown"
              value={mockPortfolioSummary.maxDrawdown}
              change={-mockPortfolioSummary.maxDrawdown}
              format="percent"
            />

            <KPICard title="Win Rate" value={mockPortfolioSummary.winRate} format="percent" />
            <KPICard title="Day Return $" value={mockPortfolioSummary.dayReturn} format="currency" />
            <KPICard title="Current Drawdown %" value={mockPortfolioSummary.drawdown} format="percent" />
            <KPICard title="Buying Power" value={mockPortfolioSummary.buyingPower} format="currency" />
            <KPICard title="Sharpe Ratio" value={mockPortfolioSummary.sharpeRatio} format="number" />
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
