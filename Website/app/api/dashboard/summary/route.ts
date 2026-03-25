import { NextResponse } from "next/server"
import { calculateMetrics } from "@/lib/alpaca"

export async function GET() {
  try {
    const metrics = await calculateMetrics()
    return NextResponse.json(metrics)
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch metrics" }, { status: 500 })
  }
}
