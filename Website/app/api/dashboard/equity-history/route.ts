import { NextRequest, NextResponse } from "next/server"
import { getPortfolioHistory } from "@/lib/alpaca"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const period = searchParams.get("period") ?? "1M"
    const timeframe = searchParams.get("timeframe") ?? "1D"

    const history = await getPortfolioHistory(period, timeframe)

    return NextResponse.json({
      timestamps: history.map((h) => h.timestamp),
      equity: history.map((h) => h.equity),
      profit_loss: history.map((h) => h.profit_loss),
      profit_loss_pct: history.map((h) => h.profit_loss_pct),
    })
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch equity history" }, { status: 500 })
  }
}
