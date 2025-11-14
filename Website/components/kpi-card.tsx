import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowUp, ArrowDown } from "lucide-react"

interface KPICardProps {
  title: string
  value: string | number
  unit?: string
  change?: number
  format?: "currency" | "percent" | "number"
  loading?: boolean
}

export function KPICard({ title, value, unit, change, format = "number", loading = false }: KPICardProps) {
  const isPositive = change === undefined ? null : change >= 0

  const formatValue = () => {
    if (loading) return "—"
    if (format === "currency") return `$${Number(value).toLocaleString()}`
    if (format === "percent") return `${Number(value).toFixed(2)}%`
    return Number(value).toFixed(2)
  }

  return (
    <Card className="bg-card border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-baseline gap-2">
          <div className="text-2xl font-bold text-foreground">{formatValue()}</div>
          {unit && <span className="text-xs text-muted-foreground">{unit}</span>}
        </div>
        {change !== undefined && (
          <div
            className={`flex items-center gap-1 text-sm font-medium ${isPositive ? "text-green-600" : "text-red-600"}`}
          >
            {isPositive ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />}
            {Math.abs(change).toFixed(2)}%
          </div>
        )}
      </CardContent>
    </Card>
  )
}
