"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { mockTrades } from "@/lib/mock-data"

export function TradesTable() {
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
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Entry Price</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Exit Price</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Realized P&L</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Return %</th>
                <th className="text-left py-2 px-2 font-medium text-muted-foreground">Date</th>
              </tr>
            </thead>
            <tbody>
              {mockTrades.map((trade, idx) => (
                <tr key={`${trade.symbol}-${idx}`} className="border-b border-border hover:bg-muted/50">
                  <td className="py-2 px-2 font-medium">{trade.symbol}</td>
                  <td className="py-2 px-2">
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium ${
                        trade.side === "BUY"
                          ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200"
                          : "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-200"
                      }`}
                    >
                      {trade.side}
                    </span>
                  </td>
                  <td className="text-right py-2 px-2">{trade.quantity}</td>
                  <td className="text-right py-2 px-2">${trade.entryPrice.toFixed(2)}</td>
                  <td className="text-right py-2 px-2">${trade.exitPrice.toFixed(2)}</td>
                  <td className={`text-right py-2 px-2 ${trade.realizedPL >= 0 ? "text-green-600" : "text-red-600"}`}>
                    ${trade.realizedPL.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td
                    className={`text-right py-2 px-2 ${trade.realizedPLPercent >= 0 ? "text-green-600" : "text-red-600"}`}
                  >
                    {trade.realizedPLPercent.toFixed(2)}%
                  </td>
                  <td className="py-2 px-2 text-muted-foreground text-xs">{trade.date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
