import Link from "next/link"

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-950 flex items-center justify-center p-6">
      <div className="text-center space-y-6 max-w-lg">
        <h1 className="text-4xl font-bold text-white">Agentic E-Trading</h1>
        <p className="text-gray-400 text-lg">
          Automated intraday equity trading powered by ML and reinforcement learning
        </p>
        <div className="flex gap-3 justify-center">
          <Link
            href="/login"
            className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-lg text-sm transition-colors"
          >
            Sign In
          </Link>
          <Link
            href="/signup"
            className="px-6 py-2.5 bg-gray-800 hover:bg-gray-700 text-white font-medium rounded-lg text-sm border border-gray-700 transition-colors"
          >
            Get Started
          </Link>
        </div>
      </div>
    </main>
  )
}
