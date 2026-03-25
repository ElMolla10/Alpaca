import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET(req: NextRequest) {
  const since = new Date()
  since.setDate(since.getDate() - 30)

  try {
    const data = await prisma.market_data.findMany({
      where: { symbol: 'AAPL', ts: { gte: since } },
      orderBy: { ts: 'asc' },
      select: { ts: true, close: true },
    })
    return NextResponse.json(data.map(d => ({ ts: d.ts, value: d.close })))
  } catch (e) {
    return NextResponse.json({ error: 'DB error' }, { status: 500 })
  }
}
