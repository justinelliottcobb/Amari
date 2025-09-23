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

### 1. Netlify (Recommended for Static Hosting)

#### Automatic Deployment
- **GitHub Integration**: Automatic deploys on push to main/master
- **Preview Deployments**: All PRs get preview URLs
- **Global CDN**: Optimized worldwide performance
- **SSL Certificates**: Automatic HTTPS setup

```bash
# Manual deployment
npm run deploy:netlify

# Preview deployment
npm run preview:netlify
```

**Setup**: See [NETLIFY_SETUP.md](./NETLIFY_SETUP.md) for complete configuration guide.

### 2. Docker (Recommended for Full-Stack Hosting)

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

### 3. Automated Deployment Script

#### Staging Environment
```bash
npm run deploy:staging
```

#### Production Environment
```bash
npm run deploy:production
```

### 4. GitHub Actions CI/CD

The repository includes automated deployment workflows:

- **Netlify Deploy**: Automatic static site deployment
- **Docker Build & Push**: Container images for infrastructure deployment
- **Test Pipeline**: Runs on all PRs and pushes
- **Preview Deployments**: Branch and PR previews

#### Workflow Triggers
- `push` to `main` → Deploy to Netlify + staging
- `push` to `master` → Deploy to Netlify + production
- `push` to `feature/api-examples-suite` → Deploy to Netlify preview
- Pull requests → Preview deployments

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

### Netlify (Static Site Hosting)
- **Automatic builds** from GitHub
- **Global CDN** with edge caching
- **Preview deployments** for PRs
- **Custom domains** with SSL
- **Form handling** and serverless functions
- **Analytics** and performance monitoring

Setup: See [NETLIFY_SETUP.md](./NETLIFY_SETUP.md)

### AWS
- ECS with Docker images
- ALB for load balancing
- CloudWatch for monitoring
- S3 + CloudFront for static hosting

### Google Cloud
- Cloud Run for serverless
- GKE for Kubernetes
- Cloud Build for CI/CD
- Firebase Hosting for static sites

### Azure
- Container Instances
- App Service
- Azure DevOps pipelines
- Static Web Apps

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