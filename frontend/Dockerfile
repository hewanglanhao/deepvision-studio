FROM node:22-alpine AS build

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY scripts ./scripts
RUN npm run setup:mode-d
COPY . .
RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/mypj-frontend/browser /usr/share/nginx/html

EXPOSE 80
