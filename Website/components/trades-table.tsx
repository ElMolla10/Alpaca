"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
interface Trade {
  symbol: string
  side: string
  qty: number
  fill_price: number
  pnl: number
  pnl_pct: number
  filled_at: string
}

export function TradesTable() {
  const [trades, setTrades] = useState<Trade[]>([])

  useEffect(() => {
    fetch("/api/dashboard/trades?days=7")
      .then((r) => r.json())
      .then((data) => Array.isArray(data) && setTrades(data))
      .catch(() => {})
  }, [])

  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <CardTitle>Recent Trades</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-2 font-medium text-muted-foreground">Symbol</th>
                <th className="text-left py-2 px-2 font-medium text-muted-foreground">Side</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Qty</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Fill Price</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">P&L</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Return %</th>
                <th className="text-left py-2 px-2 font-medium text-muted-foreground">Date</th>
              </tr>
            </thead>
            <tbody>
              {trades.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-4 text-center text-muted-foreground text-sm">No trades in the last 7 days</td>
                </tr>
              ) : (
                trades.map((trade, idx) => (
                  <tr key={`${trade.symbol}-${idx}`} className="border-b border-border hover:bg-muted/50">
                    <td className="py-2 px-2 font-medium">{trade.symbol}</td>
                    <td className="py-2 px-2">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          trade.side === "buy"
                            ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200"
                            : "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-200"
                        }`}
                      >
                        {trade.side}
                      </span>
                    </td>
                    <td className="text-right py-2 px-2">{trade.qty}</td>
                    <td className="text-right py-2 px-2">${trade.fill_price.toFixed(2)}</td>
                    <td className={`text-right py-2 px-2 ${trade.pnl >= 0 ? "text-green-600" : "text-red-600"}`}>
                      ${trade.pnl.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </td>
                    <td className={`text-right py-2 px-2 ${trade.pnl_pct >= 0 ? "text-green-600" : "text-red-600"}`}>
                      {trade.pnl_pct.toFixed(2)}%
                    </td>
                    <td className="py-2 px-2 text-muted-foreground text-xs">
                      {new Date(trade.filled_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
