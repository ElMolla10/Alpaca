"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { mockPositions } from "@/lib/mock-data"

export function PositionsTable() {
  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <CardTitle>Open Positions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-2 font-medium text-muted-foreground">Symbol</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Qty</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Entry Price</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Current Price</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Market Value</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Unrealized P&L</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Return %</th>
              </tr>
            </thead>
            <tbody>
              {mockPositions.map((pos) => (
                <tr key={pos.symbol} className="border-b border-border hover:bg-muted/50">
                  <td className="py-2 px-2 font-medium">{pos.symbol}</td>
                  <td className="text-right py-2 px-2">{pos.quantity}</td>
                  <td className="text-right py-2 px-2">${pos.entryPrice.toFixed(2)}</td>
                  <td className="text-right py-2 px-2">${pos.currentPrice.toFixed(2)}</td>
                  <td className="text-right py-2 px-2">
                    ${pos.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td className={`text-right py-2 px-2 ${pos.unrealizedPL >= 0 ? "text-green-600" : "text-red-600"}`}>
                    ${pos.unrealizedPL.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td
                    className={`text-right py-2 px-2 ${pos.unrealizedPLPercent >= 0 ? "text-green-600" : "text-red-600"}`}
                  >
                    {pos.unrealizedPLPercent.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
