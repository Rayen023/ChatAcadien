# GitHub Actions Setup for chatacadien

This document explains how to configure GitHub Actions for automatic Docker image building and deployment.

## Required GitHub Repository Secrets

To use the GitHub Actions workflow, you need to configure the following secrets in your GitHub repository:

### Docker Hub Secrets
1. **DOCKER_USERNAME** - Your Docker Hub username (rayengh01)
2. **DOCKER_PASSWORD** - Your Docker Hub password or access token

### Server Deployment Secrets (for automatic container updates)
3. **SERVER_HOST** - The IP address or hostname of your deployment server
4. **SERVER_USER** - SSH username for your server (e.g., root, ubuntu, etc.)
5. **SERVER_SSH_KEY** - Private SSH key for connecting to your server
6. **SERVER_PORT** (optional) - SSH port (defaults to 22 if not specified)

## How to Add Secrets

1. Go to your GitHub repository
2. Click on **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add each secret with the exact name and corresponding value

## Workflow Behavior

The GitHub Action will:

1. **On every push to main branch:**
   - Build the Docker image
   - Push it to Docker Hub with tags `latest` and the commit SHA
   - Deploy to your server (stop old container, pull new image, start new container)

2. **On pull requests:**
   - Only build and push the Docker image (no deployment)

## Manual Deployment

If you prefer to handle deployment manually, you can remove the "Deploy to server" step from the workflow and continue using your `updating.md` script locally.

## Alternative: GitHub Container Registry

Instead of Docker Hub, you can also use GitHub Container Registry (ghcr.io) by modifying the workflow to use `ghcr.io/${{ github.repository_owner }}/ChatchatacadienAcadien` as the image name and using `${{ secrets.GITHUB_TOKEN }}` for authentication.

## Troubleshooting

- Make sure your server has Docker installed and the user has permission to run Docker commands
- Ensure the SSH key has the correct permissions (600) on your server
- Verify that your server can access Docker Hub to pull images
- Check that all required ports (8084) are available on your server