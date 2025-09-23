# Multi-stage Dockerfile for Amari Examples Suite
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev && npm cache clean --force

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build the application
ENV NODE_ENV=production
RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 remix
RUN adduser --system --uid 1001 remix

# Copy built application
COPY --from=builder --chown=remix:remix /app/build ./build
COPY --from=builder --chown=remix:remix /app/public ./public
COPY --from=builder --chown=remix:remix /app/package.json ./package.json

USER remix

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["npm", "start"]