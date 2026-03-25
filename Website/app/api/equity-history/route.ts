import { NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"

export async function GET() {
  try {
    const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)

    const data = await prisma.market_data.findMany({
      where: {
        symbol: "AAPL",
        ts: { gte: cutoff },
      },
      orderBy: { ts: "asc" },
      select: {
        ts: true,
        close: true,
      },
    })

    return NextResponse.json(data.map((d) => ({ ts: d.ts, value: d.close })))
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch equity history" }, { status: 500 })
  }
}
