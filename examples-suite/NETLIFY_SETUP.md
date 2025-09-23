# Netlify Deployment Setup for Amari API Examples Suite

This guide covers setting up automated Netlify deployment for the Amari API Examples Suite.

## Quick Setup

### 1. Create Netlify Site

1. Go to [Netlify](https://app.netlify.com/)
2. Click "Add new site" → "Import an existing project"
3. Connect your GitHub repository
4. Select the `amari` repository
5. Configure build settings:
   - **Base directory**: `examples-suite`
   - **Build command**: `npm run build`
   - **Publish directory**: `build/client`
   - **Node version**: `18`

### 2. Set Environment Variables

In your Netlify site settings, add:
```
NODE_VERSION=18
NPM_VERSION=9
NODE_ENV=production
```

### 3. Configure GitHub Secrets

Add these secrets to your GitHub repository settings:

```bash
NETLIFY_AUTH_TOKEN=your_netlify_auth_token
NETLIFY_SITE_ID=your_netlify_site_id
```

**To get these values:**
1. **Netlify Auth Token**: Go to Netlify → User Settings → Applications → Personal Access Tokens → New Access Token
2. **Netlify Site ID**: Go to your site → Settings → General → Site Details → Site ID

## Deployment Options

### Automatic Deployment (GitHub Actions)

The included GitHub Actions workflow automatically deploys to Netlify on:
- Push to `main` branch
- Push to `master` branch
- Push to `feature/api-examples-suite` branch
- Pull requests (preview deployments)

### Manual Deployment

#### Using Netlify CLI
```bash
# Install dependencies
npm install

# Deploy to preview
npm run preview:netlify

# Deploy to production
npm run deploy:netlify
```

#### Using Netlify Dashboard
1. Go to your Netlify site dashboard
2. Click "Deploys" tab
3. Drag and drop the `build/client` folder
4. Or click "Deploy site" and select the folder

## Configuration Files

### netlify.toml
```toml
[build]
  publish = "build/client"
  command = "npm run build"

[build.environment]
  NODE_VERSION = "18"
  NPM_VERSION = "9"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

### _redirects
```
/*    /index.html   200
/docs/*  /api-reference/:splat  301
/examples/*  /:splat  301
```

## Build Process

### Local Build Testing
```bash
# Test the build locally
cd examples-suite
npm install
npm run build

# Verify build output
ls -la build/client/
```

### Build Optimization

The build is optimized for Netlify with:
- **Code splitting**: Automatic route-based splitting
- **Asset optimization**: Images, fonts, and static files
- **Cache headers**: Long-term caching for static assets
- **Compression**: Gzip enabled by default
- **Security headers**: CSP, XSS protection, etc.

## Troubleshooting

### Common Build Issues

1. **Node version mismatch**
   ```bash
   # In netlify.toml, ensure:
   NODE_VERSION = "18"
   ```

2. **Missing dependencies**
   ```bash
   # Check package.json includes all dependencies
   npm ci  # Test locally
   ```

3. **Build path issues**
   ```bash
   # Verify publish directory in netlify.toml:
   publish = "build/client"
   ```

4. **Environment variables**
   ```bash
   # Set in Netlify dashboard:
   NODE_ENV=production
   ```

### Debug Build Locally

```bash
# Clean install and build
rm -rf node_modules package-lock.json
npm install
npm run build

# Check build output
find build/client -type f -name "*.js" | head -5
find build/client -type f -name "*.css" | head -5
```

### Check Deploy Logs

1. Go to Netlify dashboard
2. Click "Deploys" tab
3. Click on a deploy to see logs
4. Look for errors in build process

## Custom Domain Setup

### Add Custom Domain
1. Go to Netlify site → Settings → Domain management
2. Click "Add custom domain"
3. Enter your domain (e.g., `api-examples.amari.dev`)
4. Follow DNS configuration instructions

### SSL Certificate
- Netlify automatically provisions SSL certificates
- HTTPS is enabled by default
- Force HTTPS redirect is recommended

## Performance Optimization

### Netlify Features
- **CDN**: Global content delivery network
- **Edge caching**: Static assets cached at edge locations
- **Image optimization**: Automatic WebP conversion
- **Brotli compression**: Better than gzip compression

### Build Performance
```bash
# Analyze bundle size
npm run build
du -sh build/client/assets/*

# Optimize if needed
# - Code splitting is already enabled
# - Tree shaking is automatic
# - Asset optimization included
```

## Monitoring and Analytics

### Netlify Analytics
1. Go to site dashboard
2. Enable Analytics (paid feature)
3. View traffic, performance metrics

### Custom Analytics
Add analytics to `app/root.tsx`:
```typescript
// Example: Google Analytics
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
```

## Security

### Headers Configuration
The `netlify.toml` includes security headers:
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- X-Content-Type-Options: nosniff
- Content-Security-Policy: Configured for app requirements

### Environment Variables
- Never commit secrets to repository
- Use Netlify environment variables for sensitive data
- GitHub secrets for deployment tokens

## Branch Deployments

### Preview Deployments
- All PRs get preview deployments automatically
- Preview URL format: `deploy-preview-{PR_NUMBER}--{SITE_NAME}.netlify.app`
- Perfect for testing before merge

### Branch Deployments
- Deploy any branch to a permanent URL
- Useful for feature branches
- Format: `{BRANCH_NAME}--{SITE_NAME}.netlify.app`

## Support and Resources

### Netlify Documentation
- [Build configuration](https://docs.netlify.com/configure-builds/overview/)
- [Deploy settings](https://docs.netlify.com/site-deploys/overview/)
- [Custom domains](https://docs.netlify.com/domains-https/custom-domains/)

### Debugging
- Check Netlify deploy logs
- Test build locally first
- Verify environment variables
- Check DNS configuration for custom domains

### Contact
For deployment issues specific to this project:
1. Check this guide
2. Review Netlify deploy logs
3. Test build locally
4. Check GitHub Actions logs