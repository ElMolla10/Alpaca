import { NextRequest, NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const symbol = searchParams.get("symbol")

    if (!symbol) {
      return NextResponse.json({ error: "symbol is required" }, { status: 400 })
    }

    const data = await prisma.feature_snapshots.findMany({
      where: { symbol },
      orderBy: { ts: "desc" },
      take: 50,
    })

    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch feature snapshots" }, { status: 500 })
  }
}
