#!/bin/bash

# Deployment script for Amari Examples Suite
set -e

echo "🚀 Starting deployment of Amari Examples Suite..."

# Configuration
ENVIRONMENT=${1:-staging}
IMAGE_TAG=${2:-latest}
COMPOSE_FILE="docker-compose.yml"

if [ "$ENVIRONMENT" = "production" ]; then
    COMPOSE_FILE="docker-compose.yml"
    echo "📦 Deploying to PRODUCTION environment"
else
    echo "📦 Deploying to STAGING environment"
fi

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if required files exist
REQUIRED_FILES=("package.json" "Dockerfile" "$COMPOSE_FILE")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file $file not found."
        exit 1
    fi
done

echo "✅ Pre-deployment checks passed"

# Build and deploy
echo "🏗️  Building application..."
docker-compose -f $COMPOSE_FILE build

echo "🔄 Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down --remove-orphans

echo "🚀 Starting new containers..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f $COMPOSE_FILE --profile production up -d
else
    docker-compose -f $COMPOSE_FILE up -d examples-suite
fi

# Health check
echo "🏥 Performing health check..."
sleep 10

MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f -s http://localhost:3000/ > /dev/null; then
        echo "✅ Health check passed! Application is running."
        break
    else
        echo "⏳ Waiting for application to start... ($((RETRY_COUNT + 1))/$MAX_RETRIES)"
        sleep 5
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Health check failed. Application may not be running properly."
    echo "📋 Container logs:"
    docker-compose -f $COMPOSE_FILE logs examples-suite
    exit 1
fi

# Post-deployment tasks
echo "🧹 Cleaning up old images..."
docker image prune -f

echo "📊 Deployment summary:"
echo "Environment: $ENVIRONMENT"
echo "Image tag: $IMAGE_TAG"
echo "Status: ✅ SUCCESS"
echo ""
echo "🌐 Application is available at:"
echo "  - Local: http://localhost:3000"
if [ "$ENVIRONMENT" = "production" ]; then
    echo "  - Public: http://localhost (via nginx)"
fi
echo ""
echo "📋 To view logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "🛑 To stop: docker-compose -f $COMPOSE_FILE down"

echo "🎉 Deployment completed successfully!"