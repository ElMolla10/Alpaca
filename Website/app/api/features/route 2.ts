import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const symbol = searchParams.get('symbol')

  if (!symbol) return NextResponse.json({ error: 'symbol required' }, { status: 400 })

  try {
    const data = await prisma.feature_snapshots.findMany({
      where: { symbol },
      orderBy: { ts: 'desc' },
      take: 50,
    })
    return NextResponse.json(data)
  } catch (e) {
    return NextResponse.json({ error: 'DB error' }, { status: 500 })
  }
}
