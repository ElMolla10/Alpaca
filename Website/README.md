# Agentic E-Trading Dashboard

A full-stack Next.js 14 application for intelligent paper trading with real-time market data from Alpaca API. Features NextAuth authentication, Prisma ORM with PlanetScale MySQL, and comprehensive trading metrics.

## Features

- **Authentication**: NextAuth with Credentials provider and PrismaAdapter
- **Database**: Prisma + PlanetScale MySQL (configurable)
- **Trading Data**: Alpaca Trading API integration (paper trading)
- **Dashboard Metrics**: 10 key performance indicators
- **Charts**: Recharts for equity visualization
- **Responsive UI**: Tailwind CSS with dark mode support
- **Mock Mode**: Runs in v0 preview without database/Alpaca keys

## Quick Start

### 1. Set Up Environment Variables

Create a `.env.local` file based on `.env.example`:

\`\`\`bash
# NextAuth
NEXTAUTH_SECRET=$(openssl rand -base64 32)
NEXTAUTH_URL=http://localhost:3000

# Database (PlanetScale)
DATABASE_URL=mysql://user:password@host/database?sslaccept=strict

# Alpaca Trading API (optional - uses mock data if not configured)
APCA_API_BASE_URL=https://paper-api.alpaca.markets
APCA_API_KEY_ID=your-alpaca-key-id
APCA_API_SECRET_KEY=your-alpaca-secret-key
\`\`\`

### 2. Install Dependencies

\`\`\`bash
npm install
# or
pnpm install
\`\`\`

### 3. Set Up Database (if using PlanetScale)

\`\`\`bash
npx prisma migrate deploy
\`\`\`

### 4. Run Development Server

\`\`\`bash
npm run dev
# or
pnpm dev
\`\`\`

Open http://localhost:3000 in your browser.

## Architecture

### Pages & Routes

- `/` - Landing page with signup/login links
- `/signup` - User registration
- `/login` - User login with NextAuth Credentials
- `/dashboard` - Protected trading dashboard

### API Routes

- `POST /api/auth/signup` - Create new user
- `GET /api/auth/[...nextauth]` - NextAuth endpoints
- `GET /api/dashboard/summary` - 10 KPI metrics
- `GET /api/dashboard/equity-history` - Portfolio equity over time
- `GET /api/dashboard/positions` - Current open positions
- `GET /api/dashboard/trades` - Recent closed trades
- `GET /api/dashboard/check-database` - Database/Alpaca config status

### Database Schema

The Prisma schema includes:
- **User** - Custom credentials auth (email, passwordHash)
- **Account, Session, VerificationToken** - NextAuth adapter models

### Alpaca Integration

- Server-only implementation (secrets never exposed to client)
- Automatic mock data fallback when API keys are missing
- Support for paper trading

## Dashboard Metrics (10 KPIs)

1. **Equity** - Total portfolio value (USD)
2. **Cash** - Available buying power (USD)
3. **Today P&L %** - Profit/loss percentage today
4. **Cumulative P&L %** - Total return since start
5. **Max Drawdown %** - Peak-to-trough decline
6. **Win Rate %** - Profitable trades / total trades
7. **Trades Today** - Number of trades executed today
8. **Avg Trade P&L %** - Average profit/loss per trade
9. **Gross Exposure %** - Total position size / equity
10. **Sharpe Ratio** - Risk-adjusted return metric

## Features Preview

- Full authentication system with database persistence
- Real-time dashboard with mock trading data
- Responsive design works on mobile, tablet, desktop
- Dark mode support
- No external dependencies required for UI

## Deployment

### Deploy to Vercel

1. Push to GitHub repository
2. Import to Vercel
3. Add environment variables in project settings
4. Deploy

The build script automatically runs `npx prisma migrate deploy` to ensure database is ready.

## Development Notes

### Mock Mode

When `DATABASE_URL` or Alpaca keys are missing:
- Authentication uses in-memory mock (demo user only)
- Dashboard displays deterministic mock data
- Banner indicates mock mode is active

### API Security

- All Alpaca API calls happen server-side only
- Session middleware protects dashboard routes
- NextAuth CSRF protection enabled
- Row-level security (RLS) ready with Supabase

## Troubleshooting

**Database Connection Error**
- Verify `DATABASE_URL` is correct
- Check PlanetScale connection is allowed from your IP
- Run `npx prisma db push` to sync schema

**Alpaca Integration Not Working**
- Confirm API keys are correct and have paper trading enabled
- Check `APCA_API_BASE_URL` points to paper trading endpoint
- Dashboard will use mock data if keys are invalid

**Session Not Persisting**
- Ensure `NEXTAUTH_SECRET` is set
- Check cookies are enabled in browser
- Verify `NEXTAUTH_URL` matches your domain
