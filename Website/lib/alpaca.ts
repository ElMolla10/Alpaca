import { getMockAccount, getMockPositions, getMockEquityHistory, getMockTrades } from "@/lib/mock-data"

export interface TradeAccount {
  equity: number
  cash: number
  buying_power: number
  portfolio_value: number
}

export interface Position {
  symbol: string
  qty: number
  side: string
  market_value: number
  unrealized_pl: number
  unrealized_plpc: number
}

export interface Trade {
  symbol: string
  side: string
  qty: number
  fill_price: number
  pnl: number
  pnl_pct: number
  filled_at: string
}

const hasCredentials =
  !!process.env.APCA_API_KEY_ID && !!process.env.APCA_API_SECRET_KEY

let alpaca: any = null

if (hasCredentials) {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Alpaca = require("@alpacahq/alpaca-trade-api")
    alpaca = new Alpaca({
      keyId: process.env.APCA_API_KEY_ID,
      secretKey: process.env.APCA_API_SECRET_KEY,
      paper: true,
    })
  } catch {
    alpaca = null
  }
}

export async function getAccount(): Promise<TradeAccount> {
  if (!alpaca) return getMockAccount()
  try {
    const account = await alpaca.getAccount()
    return {
      equity: parseFloat(account.equity),
      cash: parseFloat(account.cash),
      buying_power: parseFloat(account.buying_power),
      portfolio_value: parseFloat(account.portfolio_value),
    }
  } catch {
    return getMockAccount()
  }
}

export async function getPositions(): Promise<Position[]> {
  if (!alpaca) return getMockPositions()
  try {
    const positions = await alpaca.getPositions()
    return positions.map((p: any) => ({
      symbol: p.symbol,
      qty: parseFloat(p.qty),
      side: p.side,
      market_value: parseFloat(p.market_value),
      unrealized_pl: parseFloat(p.unrealized_pl),
      unrealized_plpc: parseFloat(p.unrealized_plpc),
    }))
  } catch {
    return getMockPositions()
  }
}

export async function getPortfolioHistory(
  period: string = "1M",
  timeframe: string = "1D"
): Promise<{ timestamp: string; equity: number; profit_loss: number; profit_loss_pct: number }[]> {
  if (!alpaca) return getMockEquityHistory(period)
  try {
    const history = await alpaca.getPortfolioHistory({ period, timeframe })
    const timestamps: number[] = history.timestamp || []
    const equities: number[] = history.equity || []
    const profitLoss: number[] = history.profit_loss || []
    const profitLossPct: number[] = history.profit_loss_pct || []

    // Use first positive equity as base
    const baseEquity = equities.find((e) => e > 0) ?? equities[0] ?? 0

    return timestamps.map((ts, i) => ({
      timestamp: new Date(ts * 1000).toISOString().split("T")[0],
      equity: equities[i] ?? 0,
      profit_loss: profitLoss[i] ?? (equities[i] - baseEquity),
      profit_loss_pct: profitLossPct[i] ?? (baseEquity > 0 ? ((equities[i] - baseEquity) / baseEquity) * 100 : 0),
    }))
  } catch {
    return getMockEquityHistory(period)
  }
}

export async function getOrders(
  status: string = "closed",
  limit: number = 50
): Promise<Trade[]> {
  if (!alpaca) return getMockTrades()
  try {
    const orders = await alpaca.getOrders({ status, limit })
    return orders.map((o: any) => ({
      symbol: o.symbol,
      side: o.side,
      qty: parseFloat(o.qty),
      fill_price: parseFloat(o.filled_avg_price || "0"),
      pnl: 0,
      pnl_pct: 0,
      filled_at: o.filled_at,
    }))
  } catch {
    return getMockTrades()
  }
}

export async function calculateMetrics() {
  const [account, positions, history, orders] = await Promise.all([
    getAccount(),
    getPositions(),
    getPortfolioHistory("1M", "1D"),
    getOrders("closed", 50),
  ])

  // Sharpe ratio
  const equities = history.map((h) => h.equity).filter((e) => e > 0)
  let sharpe = 0
  if (equities.length >= 5) {
    const returns: number[] = []
    for (let i = 1; i < equities.length; i++) {
      if (equities[i - 1] > 0) {
        returns.push((equities[i] - equities[i - 1]) / equities[i - 1])
      }
    }
    if (returns.length >= 5) {
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length
      const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length
      const stdDev = Math.sqrt(variance)
      if (stdDev > 0) {
        const annualizedReturn = mean * 252
        const annualizedStd = stdDev * Math.sqrt(252)
        sharpe = annualizedReturn / annualizedStd
      }
    }
  }

  // Max drawdown
  let maxDrawdown = 0
  let peak = equities[0] ?? 0
  for (const e of equities) {
    if (e > peak) peak = e
    if (peak > 0) {
      const dd = (peak - e) / peak
      if (dd > maxDrawdown) maxDrawdown = dd
    }
  }

  // Win rate
  const closedTrades = orders.filter((o) => o.filled_at)
  const winningTrades = closedTrades.filter((o) => o.pnl > 0)
  const winRate = closedTrades.length > 0 ? winningTrades.length / closedTrades.length : 0

  // Gross exposure
  const grossExposure = positions.reduce((sum, p) => sum + Math.abs(p.market_value), 0)

  // Trades today
  const today = new Date().toISOString().split("T")[0]
  const tradesToday = orders.filter(
    (o) => o.filled_at && o.filled_at.startsWith(today)
  ).length

  // Total P&L
  const totalPnl = positions.reduce((sum, p) => sum + p.unrealized_pl, 0)
  const totalPnlPct = account.portfolio_value > 0 ? (totalPnl / account.portfolio_value) * 100 : 0

  return {
    equity: account.equity,
    cash: account.cash,
    buying_power: account.buying_power,
    portfolio_value: account.portfolio_value,
    sharpe: Math.round(sharpe * 100) / 100,
    max_drawdown: Math.round(maxDrawdown * 10000) / 100,
    win_rate: Math.round(winRate * 10000) / 100,
    gross_exposure: Math.round(grossExposure * 100) / 100,
    trades_today: tradesToday,
    total_pnl: Math.round(totalPnl * 100) / 100,
    total_pnl_pct: Math.round(totalPnlPct * 100) / 100,
  }
}
