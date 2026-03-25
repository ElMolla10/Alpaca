import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET(req: NextRequest) {
  try {
    const data = await prisma.inference_events.findMany({
      orderBy: { ts: 'desc' },
      take: 100,
      select: { ts: true, model_id: true, request_id: true, y_pred_num: true, latency_ms: true, segment_json: true },
    })
    return NextResponse.json(data)
  } catch (e) {
    return NextResponse.json({ error: 'DB error' }, { status: 500 })
  }
}
