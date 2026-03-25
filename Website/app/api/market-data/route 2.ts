import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const symbol = searchParams.get('symbol')
  const days = parseInt(searchParams.get('days') || '30')

  if (!symbol) return NextResponse.json({ error: 'symbol required' }, { status: 400 })

  const since = new Date()
  since.setDate(since.getDate() - days)

  try {
    const data = await prisma.market_data.findMany({
      where: { symbol, ts: { gte: since } },
      orderBy: { ts: 'asc' },
      select: { ts: true, open: true, high: true, low: true, close: true, volume: true, vwap: true },
    })
    return NextResponse.json(data)
  } catch (e) {
    return NextResponse.json({ error: 'DB error' }, { status: 500 })
  }
}
