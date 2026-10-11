# Phone page (Next.js), production build.
FROM node:22-slim
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# next.config.ts reads this at build time for the /api/* forwarding, so it is set before `next build`.
ARG PILLTALLY_API=http://api:8000
ENV PILLTALLY_API=$PILLTALLY_API NEXT_TELEMETRY_DISABLED=1
RUN npm run build
EXPOSE 3000
CMD ["npm", "start", "--", "-H", "0.0.0.0", "-p", "3000"]
