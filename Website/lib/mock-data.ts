export function getMockAccount() {
  return {
    equity: 100000,
    cash: 25000,
    buying_power: 100000,
    portfolio_value: 100000,
  }
}

export function getMockPositions() {
  return [
    {
      symbol: "AAPL",
      qty: 50,
      side: "long",
      market_value: 9250.0,
      unrealized_pl: 250.0,
      unrealized_plpc: 0.0278,
    },
    {
      symbol: "MSFT",
      qty: 30,
      side: "long",
      market_value: 12450.0,
      unrealized_pl: 450.0,
      unrealized_plpc: 0.0375,
    },
    {
      symbol: "TSLA",
      qty: 20,
      side: "short",
      market_value: 4800.0,
      unrealized_pl: -120.0,
      unrealized_plpc: -0.0244,
    },
  ]
}

export function getMockEquityHistory(period: string = "1M") {
  const periodDays: Record<string, number> = {
    "1D": 1,
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "1A": 365,
    "all": 365,
  }
  const days = periodDays[period] ?? 30
  const result = []
  let equity = 95000
  const now = Date.now()

  for (let i = days; i >= 0; i--) {
    const ts = new Date(now - i * 24 * 60 * 60 * 1000).toISOString().split("T")[0]
    equity = equity + (Math.random() - 0.48) * 800
    result.push({
      timestamp: ts,
      equity: Math.round(equity * 100) / 100,
      profit_loss: Math.round((equity - 95000) * 100) / 100,
      profit_loss_pct: Math.round(((equity - 95000) / 95000) * 10000) / 100,
    })
  }
  return result
}

export function getMockTrades() {
  return [
    {
      symbol: "AAPL",
      side: "buy",
      qty: 10,
      fill_price: 182.5,
      pnl: 125.0,
      pnl_pct: 0.685,
      filled_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
    },
    {
      symbol: "MSFT",
      side: "sell",
      qty: 5,
      fill_price: 415.0,
      pnl: -45.0,
      pnl_pct: -0.217,
      filled_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    },
    {
      symbol: "NVDA",
      side: "buy",
      qty: 8,
      fill_price: 875.0,
      pnl: 320.0,
      pnl_pct: 0.457,
      filled_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    },
  ]
}

// ---------------------------------------------------------------------------
// Named constant exports consumed by existing UI components
// ---------------------------------------------------------------------------

export const mockPortfolioSummary = {
  totalEquity: 100000,
  cash: 25000,
  buyingPower: 100000,
  portfolioValue: 100000,
  dayReturn: 580,
  dayReturnPercentage: 0.58,
  returnPercentage: 5.26,
  maxDrawdown: 3.14,
  drawdown: 1.2,
  winRate: 62.5,
  sharpeRatio: 1.42,
}

// Stable 30-day equity walk (seeded so values don't change between renders)
function generateEquityHistory() {
  const result: { date: string; equity: number }[] = []
  let equity = 95000
  const now = new Date("2026-03-25").getTime()
  for (let i = 30; i >= 0; i--) {
    const date = new Date(now - i * 24 * 60 * 60 * 1000).toISOString().split("T")[0]
    // Deterministic pseudo-random walk using sine
    equity = equity + Math.sin(i * 1.7 + 0.3) * 600 + 120
    result.push({ date, equity: Math.round(equity * 100) / 100 })
  }
  return result
}

export const mockEquityHistory = generateEquityHistory()

export const mockPositions = [
  { symbol: "AAPL", qty: 50, side: "long", market_value: 9250.0, unrealized_pl: 250.0, unrealized_plpc: 0.0278 },
  { symbol: "MSFT", qty: 30, side: "long", market_value: 12450.0, unrealized_pl: 450.0, unrealized_plpc: 0.0375 },
  { symbol: "TSLA", qty: 20, side: "short", market_value: 4800.0, unrealized_pl: -120.0, unrealized_plpc: -0.0244 },
]

export const mockTrades = [
  { symbol: "AAPL", side: "buy", qty: 10, fill_price: 182.5, pnl: 125.0, pnl_pct: 0.685, filled_at: "2026-03-24T14:32:00Z" },
  { symbol: "MSFT", side: "sell", qty: 5, fill_price: 415.0, pnl: -45.0, pnl_pct: -0.217, filled_at: "2026-03-23T10:15:00Z" },
  { symbol: "NVDA", side: "buy", qty: 8, fill_price: 875.0, pnl: 320.0, pnl_pct: 0.457, filled_at: "2026-03-22T09:48:00Z" },
]
