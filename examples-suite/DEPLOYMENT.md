# Amari Examples Suite - Deployment Guide

This guide covers deployment options for the Amari Mathematical Computing Library API Examples Suite.

## Quick Start

### Local Development
```bash
npm run dev
```

### Local Docker Deployment
```bash
npm run deploy:local
```

## Deployment Options

### 1. Docker (Recommended)

#### Single Container
```bash
# Build the image
npm run docker:build

# Run the container
npm run docker:run
```

#### Docker Compose (with Nginx)
```bash
# Start services
docker-compose up -d

# Production mode with Nginx proxy
docker-compose --profile production up -d

# Stop services
npm run deploy:stop
```

### 2. Automated Deployment Script

#### Staging Environment
```bash
npm run deploy:staging
```

#### Production Environment
```bash
npm run deploy:production
```

### 3. GitHub Actions CI/CD

The repository includes automated deployment workflows:

- **Test Pipeline**: Runs on all PRs and pushes
- **Build & Push**: Creates Docker images for main/master branches
- **Deploy**: Automatically deploys to staging/production

#### Workflow Triggers
- `push` to `main` → Deploy to staging
- `push` to `master` → Deploy to production
- Manual trigger via `workflow_dispatch`

## Configuration

### Environment Variables

```bash
NODE_ENV=production
PORT=3000
HOSTNAME=0.0.0.0
```

### Docker Configuration

#### Dockerfile Features
- Multi-stage build for optimized image size
- Non-root user for security
- Health checks included
- Production-ready Node.js setup

#### Docker Compose Features
- Nginx reverse proxy (production profile)
- Health checks
- Volume persistence
- Security headers

### Nginx Configuration

The included `nginx.conf` provides:
- Reverse proxy to Remix app
- Static asset caching
- Security headers
- Gzip compression
- Health check endpoint

## Health Checks

### Application Health Check
```bash
npm run health
```

### Docker Health Check
```bash
docker-compose ps
```

### Manual Health Check
```bash
curl http://localhost:3000/
```

## Performance Optimizations

### Build Optimizations
- Tree shaking enabled
- Code splitting
- Asset optimization
- Bundle analysis

### Runtime Optimizations
- Nginx static asset caching
- Gzip compression
- Security headers
- Connection keep-alive

## Security Features

### Container Security
- Non-root user execution
- Minimal base image (Alpine)
- No unnecessary packages
- Security scanning ready

### Web Security
- HTTPS ready (certificates in `/ssl`)
- Security headers (CSP, HSTS, etc.)
- XSS protection
- CSRF protection

## Monitoring & Logging

### Application Logs
```bash
# Docker Compose logs
docker-compose logs -f examples-suite

# Container logs
docker logs amari-examples-suite
```

### Health Monitoring
- Built-in health check endpoints
- Container health status
- Application performance metrics

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   docker-compose down
   # or change port in docker-compose.yml
   ```

2. **Build Failures**
   ```bash
   # Clean build
   docker system prune -f
   npm run docker:build
   ```

3. **Health Check Failures**
   ```bash
   # Check logs
   docker-compose logs examples-suite

   # Restart services
   docker-compose restart
   ```

### Debug Mode
```bash
# Run with debug output
DEBUG=* npm run dev

# Docker debug
docker-compose logs -f
```

## Scaling & Production

### Horizontal Scaling
```bash
# Scale containers
docker-compose up -d --scale examples-suite=3
```

### Load Balancing
- Nginx included for basic load balancing
- Ready for external load balancers (AWS ALB, etc.)

### Database Integration
- Stateless application design
- Ready for external databases
- Session storage configurable

## Deployment Checklist

### Pre-deployment
- [ ] Run tests: `npm test`
- [ ] Type check: `npm run typecheck`
- [ ] Build check: `npm run build`
- [ ] Security scan: `docker scan`

### Post-deployment
- [ ] Health check: `npm run health`
- [ ] Performance test
- [ ] Log monitoring
- [ ] Error tracking

## Platform-Specific Deployment

### AWS
- ECS with Docker images
- ALB for load balancing
- CloudWatch for monitoring

### Google Cloud
- Cloud Run for serverless
- GKE for Kubernetes
- Cloud Build for CI/CD

### Azure
- Container Instances
- App Service
- Azure DevOps pipelines

### Kubernetes
```yaml
# Ready for K8s deployment
# See k8s/ directory for manifests
```

## Support

For deployment issues:
1. Check this guide
2. Review application logs
3. Verify system requirements
4. Contact development team

## Version History

- v1.0.0 - Initial deployment configuration
- Docker support with multi-stage builds
- CI/CD pipeline with GitHub Actions
- Production-ready Nginx configuration