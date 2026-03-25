import { NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"

export async function GET() {
  try {
    const data = await prisma.inference_events.findMany({
      orderBy: { ts: "desc" },
      take: 100,
      select: {
        ts: true,
        model_id: true,
        request_id: true,
        y_pred_num: true,
        latency_ms: true,
        segment_json: true,
      },
    })

    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: "Failed to fetch inference events" }, { status: 500 })
  }
}
