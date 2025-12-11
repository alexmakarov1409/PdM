#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROD_COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"
BACKUP_DIR="backups"
LOG_DIR="logs"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    success "Docker and Docker Compose are installed"
}

# Check if .env file exists
check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        warning ".env file not found. Creating from example..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            warning "Please edit .env file with your configuration"
            read -p "Press Enter to continue or Ctrl+C to abort..."
        else
            error ".env.example file not found"
            exit 1
        fi
    fi
    
    success "Environment file checked"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p $BACKUP_DIR
    mkdir -p $LOG_DIR
    mkdir -p models
    mkdir -p data
    
    success "Directories created"
}

# Backup existing data
backup_data() {
    log "Creating backup of existing data..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"
    
    # Backup database
    if docker-compose ps postgres | grep -q "Up"; then
        log "Backing up PostgreSQL database..."
        docker-compose exec -T postgres pg_dumpall -U postgres > "$BACKUP_DIR/database_$TIMESTAMP.sql"
        success "Database backed up"
    fi
    
    # Backup volumes
    if [ -d "models" ] && [ "$(ls -A models 2>/dev/null)" ]; then
        log "Backing up models..."
        tar -czf "$BACKUP_DIR/models_$TIMESTAMP.tar.gz" models/
        success "Models backed up"
    fi
    
    if [ -d "data" ] && [ "$(ls -A data 2>/dev/null)" ]; then
        log "Backing up data..."
        tar -czf "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" data/
        success "Data backed up"
    fi
    
    success "Backup completed: $BACKUP_FILE"
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    
    docker-compose pull
    
    success "Images pulled"
}

# Build images
build_images() {
    log "Building Docker images..."
    
    if [ "$1" = "production" ]; then
        docker-compose -f $COMPOSE_FILE -f $PROD_COMPOSE_FILE build --parallel
    else
        docker-compose build --parallel
    fi
    
    success "Images built"
}

# Start services
start_services() {
    log "Starting services..."
    
    if [ "$1" = "production" ]; then
        docker-compose -f $COMPOSE_FILE -f $PROD_COMPOSE_FILE up -d
    else
        docker-compose up -d
    fi
    
    success "Services started"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    log "Waiting for PostgreSQL..."
    until docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; do
        sleep 2
    done
    success "PostgreSQL is ready"
    
    # Wait for Redis
    log "Waiting for Redis..."
    until docker-compose exec -T redis redis-cli ping | grep -q "PONG"; do
        sleep 2
    done
    success "Redis is ready"
    
    # Wait for API
    log "Waiting for API..."
    until curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; do
        sleep 2
    done
    success "API is ready"
    
    success "All services are ready"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    docker-compose exec -T api python -c "
from alembic.config import Config
from alembic import command
alembic_cfg = Config('alembic.ini')
command.upgrade(alembic_cfg, 'head')
"
    
    success "Migrations completed"
}

# Run tests
run_tests() {
    if [ "$1" != "production" ]; then
        log "Running tests..."
        
        docker-compose exec -T api pytest /app/tests -v
        
        success "Tests completed"
    fi
}

# Show deployment status
show_status() {
    log "Deployment status:"
    echo ""
    
    # Show running containers
    docker-compose ps
    
    echo ""
    log "Service URLs:"
    echo "  API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  Grafana: http://localhost:3000"
    echo "  Prometheus: http://localhost:9090"
    echo "  Kafka UI: http://localhost:8080"
    echo ""
    
    # Check API health
    if curl -s http://localhost:8000/api/v1/health | grep -q "healthy"; then
        success "API is healthy"
    else
        warning "API health check failed"
    fi
}

# Cleanup old backups
cleanup_backups() {
    log "Cleaning up old backups (keeping last 10)..."
    
    # Keep last 10 backups
    ls -t $BACKUP_DIR/*.sql 2>/dev/null | tail -n +11 | xargs -r rm
    ls -t $BACKUP_DIR/*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm
    
    success "Old backups cleaned"
}

# Main deployment function
deploy() {
    MODE=${1:-development}
    
    log "Starting deployment in $MODE mode..."
    
    # Step 1: Check prerequisites
    check_docker
    check_env
    
    # Step 2: Prepare directories
    create_directories
    
    # Step 3: Backup existing data
    backup_data
    
    # Step 4: Pull and build images
    pull_images
    build_images $MODE
    
    # Step 5: Start services
    start_services $MODE
    
    # Step 6: Wait for services
    wait_for_services
    
    # Step 7: Run migrations
    run_migrations
    
    # Step 8: Run tests (development only)
    run_tests $MODE
    
    # Step 9: Cleanup
    cleanup_backups
    
    success "Deployment completed successfully!"
    
    # Step 10: Show status
    show_status
}

# Rollback function
rollback() {
    if [ -z "$1" ]; then
        error "Please specify backup timestamp to rollback to"
        echo "Usage: $0 rollback <timestamp>"
        echo "Available backups:"
        ls -la $BACKUP_DIR/*.sql 2>/dev/null | awk '{print $9}' | sed 's/.*backup_//' | sed 's/\.sql//'
        exit 1
    fi
    
    TIMESTAMP=$1
    BACKUP_FILE="$BACKUP_DIR/database_$TIMESTAMP.sql"
    
    if [ ! -f "$BACKUP_FILE" ]; then
        error "Backup file not found: $BACKUP_FILE"
        exit 1
    fi
    
    log "Rolling back to backup: $TIMESTAMP"
    
    # Stop services
    docker-compose down
    
    # Restore database
    log "Restoring database..."
    docker-compose up -d postgres
    sleep 10
    docker-compose exec -T postgres psql -U postgres -d predictive_maintenance < "$BACKUP_FILE"
    
    # Start services
    docker-compose up -d
    
    success "Rollback completed"
}

# Main script
case "$1" in
    "deploy")
        deploy "${2:-development}"
        ;;
    "production")
        deploy "production"
        ;;
    "rollback")
        rollback "$2"
        ;;
    "status")
        show_status
        ;;
    "backup")
        backup_data
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "stop")
        docker-compose down
        ;;
    "start")
        docker-compose up -d
        ;;
    "restart")
        docker-compose restart
        ;;
    "clean")
        docker-compose down -v
        ;;
    *)
        echo "Usage: $0 {deploy|production|rollback|status|backup|logs|stop|start|restart|clean}"
        echo ""
        echo "Commands:"
        echo "  deploy [development|staging]  Deploy the application"
        echo "  production                    Deploy in production mode"
        echo "  rollback <timestamp>          Rollback to a specific backup"
        echo "  status                        Show deployment status"
        echo "  backup                        Create a backup"
        echo "  logs                          Show logs"
        echo "  stop                          Stop services"
        echo "  start                         Start services"
        echo "  restart                       Restart services"
        echo "  clean                         Stop and remove volumes"
        exit 1
        ;;
esac