"use client"

import { useState, useEffect } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
export function EquityChart() {
  const [chartData, setChartData] = useState<{ date: string; equity: number }[]>([])

  useEffect(() => {
    fetch("/api/dashboard/equity-history?period=1M&timeframe=1D")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.timestamps) && data.timestamps.length > 0) {
          setChartData(
            data.timestamps.map((date: string, i: number) => ({
              date,
              equity: Math.round(data.equity[i] ?? 0),
            }))
          )
        }
      })
      .catch(() => {})
  }, [])

  return (
    <Card className="col-span-full bg-card border-border">
      <CardHeader>
        <CardTitle>Portfolio Equity</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="date" stroke="var(--color-muted-foreground)" style={{ fontSize: "12px" }} />
            <YAxis stroke="var(--color-muted-foreground)" style={{ fontSize: "12px" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-card)",
                border: `1px solid var(--color-border)`,
                borderRadius: "8px",
              }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(value: any) => `$${Number(value ?? 0).toLocaleString()}`}
            />
            <Line
              type="monotone"
              dataKey="equity"
              stroke="var(--color-primary)"
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
