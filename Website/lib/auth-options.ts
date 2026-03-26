import CredentialsProvider from "next-auth/providers/credentials"
import { prisma } from "@/lib/prisma"
import bcrypt from "bcryptjs"
import type { NextAuthOptions } from "next-auth"

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          throw new Error("Email and password required")
        }
        const user = await prisma.user.findUnique({ where: { email: credentials.email.toLowerCase() } })
        if (!user) throw new Error("No user found with this email")
        const passwordMatch = await bcrypt.compare(credentials.password, user.passwordHash)
        if (!passwordMatch) throw new Error("Invalid password")
        return { id: user.id, email: user.email }
      },
    }),
  ],
  session: { strategy: "jwt" as const },
  pages: { signIn: "/login", error: "/login" },
  callbacks: {
    async jwt({ token, user }: { token: any; user: any }) {
      if (user) token.id = user.id
      return token
    },
    async session({ session, token }: { session: any; token: any }) {
      if (session.user) session.user.id = token.id
      return session
    },
  },
  secret: process.env.NEXTAUTH_SECRET,
}
