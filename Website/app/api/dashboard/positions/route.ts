import { NextResponse } from "next/server"
import { getPositions } from "@/lib/alpaca"

export async function GET() {
  try {
    const positions = await getPositions()
    return NextResponse.json(positions)
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch positions" }, { status: 500 })
  }
}
