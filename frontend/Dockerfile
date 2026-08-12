FROM node:20-slim

WORKDIR /app

# Lockfile-first so a source change does not reinstall node_modules.
COPY package.json package-lock.json* ./
RUN npm install

COPY . .

EXPOSE 5173
# Dev server rather than a static build behind nginx: this is a research
# artifact meant to be read and modified, and one fewer moving part matters more
# than production asset serving.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
