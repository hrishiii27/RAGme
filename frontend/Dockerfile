# Stage 1: Build the React application
FROM node:20-alpine AS build

WORKDIR /app

# Copy dependency definitions and install
COPY package.json ./
# Install dependencies cleanly
RUN npm install

# Copy source code and build
COPY . .
RUN npm run build

# Stage 2: Serve optimized assets with Nginx
FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled SPA static assets from builder stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
