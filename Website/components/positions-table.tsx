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
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Side</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Market Value</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Unrealized P&L</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Return %</th>
              </tr>
            </thead>
            <tbody>
              {mockPositions.map((pos) => (
                <tr key={pos.symbol} className="border-b border-border hover:bg-muted/50">
                  <td className="py-2 px-2 font-medium">{pos.symbol}</td>
                  <td className="text-right py-2 px-2">{pos.qty}</td>
                  <td className="text-right py-2 px-2">{pos.side}</td>
                  <td className="text-right py-2 px-2">
                    ${pos.market_value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td className={`text-right py-2 px-2 ${pos.unrealized_pl >= 0 ? "text-green-600" : "text-red-600"}`}>
                    ${pos.unrealized_pl.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                  <td
                    className={`text-right py-2 px-2 ${pos.unrealized_plpc >= 0 ? "text-green-600" : "text-red-600"}`}
                  >
                    {(pos.unrealized_plpc * 100).toFixed(2)}%
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
