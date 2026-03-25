import { NextRequest, NextResponse } from "next/server"
import { getOrders } from "@/lib/alpaca"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const days = parseInt(searchParams.get("days") ?? "7", 10)

    const orders = await getOrders("closed", 50)

    const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString()
    const filtered = orders.filter((o) => o.filled_at && o.filled_at >= cutoff)

    return NextResponse.json(filtered)
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch trades" }, { status: 500 })
  }
}
