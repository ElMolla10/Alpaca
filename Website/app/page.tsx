import Link from "next/link"

export default function HomePage() {
  return (
    <main className="min-h-screen bg-gray-950 flex flex-col items-center justify-center px-4">
      <div className="text-center max-w-2xl">
        <h1 className="text-5xl font-bold text-white mb-4">Agentic E-Trading</h1>
        <p className="text-lg text-gray-400 mb-10">
          Automated intraday equity trading powered by ML and reinforcement learning
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            href="/login"
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors"
          >
            Sign In
          </Link>
          <Link
            href="/signup"
            className="px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white font-semibold rounded-lg transition-colors"
          >
            Get Started
          </Link>
        </div>
      </div>
    </main>
  )
}
