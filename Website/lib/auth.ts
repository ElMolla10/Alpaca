import { getServerSession } from "next-auth/next"
import { authOptions } from "@/lib/auth-options"
import { redirect } from "next/navigation"

export async function getSession() {
  return getServerSession(authOptions)
}

export async function protectRoute() {
  const session = await getSession()
  if (!session?.user) {
    redirect("/login")
  }
  return session
}
