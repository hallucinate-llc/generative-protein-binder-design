FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat

# The build context is the repo root so bring the local SDK and dashboard manifests into the image
COPY mcp-js-sdk /mcp-js-sdk
WORKDIR /app
COPY mcp-dashboard/package*.json ./
RUN npm install \
	&& rm -rf node_modules/@generative-protein/mcp-js-sdk \
	&& cp -r /mcp-js-sdk node_modules/@generative-protein/mcp-js-sdk

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /mcp-js-sdk /mcp-js-sdk
COPY mcp-dashboard .
RUN mkdir -p public

# Disable telemetry during build
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Automatically leverage output traces to reduce image size
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
