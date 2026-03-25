import { NextRequest, NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const symbol = searchParams.get("symbol")
    const days = parseInt(searchParams.get("days") ?? "30", 10)

    if (!symbol) {
      return NextResponse.json({ error: "symbol is required" }, { status: 400 })
    }

    const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000)

    const data = await prisma.market_data.findMany({
      where: {
        symbol,
        ts: { gte: cutoff },
      },
      orderBy: { ts: "asc" },
      select: {
        ts: true,
        open: true,
        high: true,
        low: true,
        close: true,
        volume: true,
        vwap: true,
      },
    })

    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch market data" }, { status: 500 })
  }
}
