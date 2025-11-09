# ML Training Project Boilerplate

A comprehensive boilerplate template for machine learning training projects that demonstrates:
- Setting up PostgreSQL using Docker Compose
- Training models with PyTorch (includes MNIST example)
- Storing training results in PostgreSQL
- Using Alembic for database migrations
- Following software engineering best practices (SRP, Dependency Injection)

**This is a flexible template** - you can easily modify it to train any model with any dataset. The current implementation includes an MNIST example, but the architecture is designed to be adaptable to your specific use case.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Architecture Overview](#architecture-overview)
- [Key Concepts](#key-concepts)
- [Configuration](#configuration)
- [Database Schema](#database-schema)
- [Usage Examples](#usage-examples)
- [Customizing for Your Project](#customizing-for-your-project)
- [Customizing Database Schema and Alembic Migrations](#customizing-database-schema-and-alembic-migrations)
- [Quick Reference](#quick-reference)
- [Troubleshooting](#troubleshooting)
- [Testing the Setup](#testing-the-setup)
- [FAQ](#faq)
- [Code Quality and Formatting](#code-quality-and-formatting)
- [CI/CD Pipeline](#cicd-pipeline)
- [Next Steps](#next-steps)
- [Resources](#resources)

## Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.9+** (Python 3.10-3.13 recommended for best compatibility) - [Download Python](https://www.python.org/downloads/)
- **Docker** and **Docker Compose** - [Install Docker](https://docs.docker.com/get-docker/)
- **Git** (optional) - [Install Git](https://git-scm.com/downloads)

> **Note**: If using Python 3.14+, some dependencies may need to build from source. The project uses flexible version requirements (`>=`) to ensure compatibility across Python versions.

## Project Structure

```
final-project-template/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── main.py                # Application entry point
│   ├── ci_train.py            # CI/CD training script (no DB)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── base.py            # Database connection setup
│   │   ├── models.py          # SQLAlchemy models
│   │   └── repository.py      # Repository pattern implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── mnist_model.py     # Example PyTorch model (MNIST) - replace with your model
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Training logic (customize for your dataset)
│   └── services/
│       ├── __init__.py
│       └── training_service.py # Service layer
├── migrations/                # Alembic migrations
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 001_initial.py
├── docker-compose.yml         # PostgreSQL container setup
├── alembic.ini                # Alembic configuration
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies (pre-commit, ruff)
├── pyproject.toml             # Ruff configuration
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD pipeline
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Setup Instructions

### 1. Clone or Download the Project

```bash
cd /path/to/your/workspace
# If using git:
git clone <repository-url>
cd final-project-template
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Upgrade pip first (recommended)
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Expected output**: All packages should install successfully. The installation may take a few minutes, especially for PyTorch.

> **Tip**: If you encounter build errors, ensure you have the latest pip version and try installing packages individually to identify the problematic dependency.

**Optional: Install Development Dependencies (for pre-commit hooks)**

```bash
# Install development dependencies (pre-commit, ruff)
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

This will set up automatic code formatting and linting using ruff. The hooks will run automatically on every commit, ensuring code quality and consistent formatting.

### 4. Set Up Environment Variables

Create a `.env` file in the project root. The project will work with default values, but you can create a `.env` file to customize settings:

```bash
# Create .env file (optional - defaults will be used if not present)
touch .env
```

Add the following to `.env` (or leave empty to use defaults):

```env
# Database Configuration (defaults shown)
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mnist_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mnist_db

# Training Configuration (optional)
TRAINING_BATCH_SIZE=64
TRAINING_LEARNING_RATE=0.001
TRAINING_EPOCHS=5
TRAINING_DEVICE=cpu
```

> **Note**: The `.env` file is optional. The application will use sensible defaults if the file doesn't exist. However, creating it allows you to customize settings easily.

### 5. Start PostgreSQL with Docker Compose

**Important**: Make sure Docker Desktop (or Docker daemon) is running before proceeding.

```bash
# Start PostgreSQL container in detached mode
docker-compose up -d
```

This will:
- Pull the PostgreSQL 15 Alpine image (if not already present)
- Start a PostgreSQL container named `mnist_postgres`
- Create the database `mnist_db`
- Expose PostgreSQL on port 5432
- Create a persistent volume for data

**Verify the container is running:**

```bash
docker-compose ps
```

You should see output like:
```
NAME             IMAGE                STATUS
mnist_postgres   postgres:15-alpine   Up (health: starting)
```

Wait a few seconds for PostgreSQL to fully start, then check again - the health status should show "healthy".

**Check container logs if needed:**
```bash
docker-compose logs postgres
```

### 6. Run Database Migrations

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run migrations to create tables
alembic upgrade head
```

**Expected output:**
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 001_initial, Initial migration
```

This will create the `training_runs` and `epoch_results` tables in your database.

**Verify tables were created:**
```bash
docker exec mnist_postgres psql -U postgres -d mnist_db -c "\dt"
```

You should see both `training_runs` and `epoch_results` tables listed.

## Running the Project

### Quick Start

Once everything is set up, run:

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the training
python -m src.main
```

### What to Expect

When running the example (MNIST), the script will:

1. **Download dataset** (first run only, ~60MB for MNIST)
   - Progress bars will show download progress
   - Dataset is saved to `./data/` directory

2. **Train the model** for the specified number of epochs (default: 5)
   - You'll see progress for each epoch:
     ```
     Epoch 1/5 - Train Loss: 0.2187, Train Acc: 93.34% - Val Loss: 0.0436, Val Acc: 98.51%
     Epoch 2/5 - Train Loss: 0.0830, Train Acc: 97.50% - Val Loss: 0.0358, Val Acc: 98.76%
     ...
     ```

3. **Save results to PostgreSQL**
   - Training run metadata is saved
   - All epoch results are saved

4. **Display summary**:
   ```
   Training completed successfully!
   Training Run ID: 1
   Final Accuracy: 99.06%
   Final Loss: 0.0272
   ```

**Expected training time**: ~2-3 minutes on CPU for 5 epochs with MNIST (varies by model and hardware).

> **Note**: This boilerplate includes an MNIST example, but you can modify `src/models/`, `src/training/trainer.py`, and `src/main.py` to train any model with any dataset.

### Train the Model and Save Results

```bash
python -m src.main
```

This will:
1. Download the MNIST dataset (if not already present)
2. Train the model for the specified number of epochs
3. Save training results to PostgreSQL
4. Display a summary of the training run

### Verify Data in Database

**Option 1: Using Docker exec (Command Line)**

```bash
# Query training runs
docker exec mnist_postgres psql -U postgres -d mnist_db -c "SELECT id, model_name, epochs, final_accuracy, final_loss FROM training_runs;"

# Query epoch results
docker exec mnist_postgres psql -U postgres -d mnist_db -c "SELECT training_run_id, epoch, train_accuracy, val_accuracy FROM epoch_results ORDER BY training_run_id, epoch;"
```

**Option 2: Interactive PostgreSQL Shell**

```bash
# Connect to PostgreSQL container
docker exec -it mnist_postgres psql -U postgres -d mnist_db

# Then run SQL queries:
SELECT * FROM training_runs;
SELECT * FROM epoch_results ORDER BY training_run_id, epoch;
\q  # Exit
```

**Option 3: Database GUI Tools**

Use a database GUI tool like pgAdmin, DBeaver, or TablePlus with these connection settings:

- **Host**: `localhost`
- **Port**: `5432`
- **Database**: `mnist_db`
- **Username**: `postgres`
- **Password**: `postgres`

**Expected Query Results:**

After running training, you should see:
- 1 row in `training_runs` table with your training metadata
- 5 rows in `epoch_results` table (one per epoch) with detailed metrics

## Architecture Overview

This project follows several software engineering best practices:

### Single Responsibility Principle (SRP)

Each module has a single, well-defined responsibility:

- **`config.py`**: Configuration management only
- **`database/base.py`**: Database connection management
- **`database/models.py`**: Data models only
- **`database/repository.py`**: Data access operations
- **`models/mnist_model.py`**: Model architecture definition (example - replace with your model)
- **`training/trainer.py`**: Training logic only (customize for your dataset)
- **`services/training_service.py`**: Orchestration of training and persistence

### Dependency Injection

Dependencies are injected rather than created within classes:

- `Database` receives `DatabaseConfig` via constructor
- `TrainingService` receives `MNISTTrainer` and repositories via constructor
- `Repositories` receive `Session` via constructor
- Factory functions (`get_database()`, `create_training_service()`) handle dependency wiring

### Repository Pattern

Data access is abstracted through repositories:
- `TrainingRunRepository`: Manages training run records
- `EpochResultRepository`: Manages epoch result records

This makes the code testable and allows easy swapping of data sources.

### Service Layer

The `TrainingService` coordinates between:
- Training logic (`MNISTTrainer`)
- Data persistence (`Repositories`)

This keeps business logic separate from data access.

## Key Concepts

### Docker Compose

Docker Compose allows you to define and run multi-container Docker applications. Our `docker-compose.yml` defines:
- PostgreSQL service with environment variables
- Volume for data persistence
- Health checks for reliability

### Alembic Migrations

Alembic is a database migration tool for SQLAlchemy. It allows you to:
- Version control your database schema
- Apply incremental changes
- Rollback changes if needed

For detailed migration instructions, see the [Customizing Database Schema and Alembic Migrations](#customizing-database-schema-and-alembic-migrations) section.

### SQLAlchemy ORM

SQLAlchemy provides:
- Object-Relational Mapping (ORM) for Python
- Database abstraction layer
- Session management
- Query building

### PyTorch Training

The training process (example with MNIST):
1. Loads dataset (MNIST in the example)
2. Creates data loaders with batching
3. Trains model epoch by epoch
4. Validates after each epoch
5. Tracks metrics (loss, accuracy)

You can modify `src/training/trainer.py` to work with any dataset and model architecture.

## Configuration

### Database Configuration

Configured in `src/config.py` via `DatabaseConfig`:
- Reads from `.env` file
- Supports `POSTGRES_*` prefix variables
- Constructs database URL automatically

### Training Configuration

Configured in `src/config.py` via `TrainingConfig`:
- Batch size: Number of samples per batch
- Learning rate: Step size for optimization
- Epochs: Number of training iterations
- Device: `cpu` or `cuda` (for GPU)

## Database Schema

### `training_runs` Table

Stores metadata about each training run:

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `model_name` | String(100) | Name of the model |
| `epochs` | Integer | Number of epochs trained |
| `batch_size` | Integer | Batch size used |
| `learning_rate` | Float | Learning rate used |
| `final_loss` | Float | Final validation loss |
| `final_accuracy` | Float | Final validation accuracy (%) |
| `training_time_seconds` | Float | Total training time |
| `created_at` | DateTime | Timestamp of creation |
| `notes` | Text | Optional notes |

### `epoch_results` Table

Stores per-epoch metrics:

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `training_run_id` | Integer | Foreign key to training_runs |
| `epoch` | Integer | Epoch number |
| `train_loss` | Float | Training loss |
| `train_accuracy` | Float | Training accuracy (%) |
| `val_loss` | Float | Validation loss |
| `val_accuracy` | Float | Validation accuracy (%) |
| `created_at` | DateTime | Timestamp of creation |

## Usage Examples

### Custom Training Configuration

Modify `.env` to change training parameters:

```env
TRAINING_BATCH_SIZE=128
TRAINING_LEARNING_RATE=0.0001
TRAINING_EPOCHS=10
TRAINING_DEVICE=cpu
```

### Query Training Results Programmatically

```python
from src.database.base import get_database
from src.database.repository import TrainingRunRepository

db = get_database()
session = next(db.get_session())

repo = TrainingRunRepository(session)
runs = repo.get_all()

for run in runs:
    print(f"Run {run.id}: {run.final_accuracy:.2f}% accuracy")
    print(f"  Model: {run.model_name}")
    print(f"  Epochs: {run.epochs}")
    print(f"  Training time: {run.training_time_seconds:.2f}s")
```

### Accessing Epoch-Level Results

```python
from src.database.repository import EpochResultRepository

epoch_repo = EpochResultRepository(session)
epochs = epoch_repo.get_by_training_run_id(run_id=1)

for epoch in epochs:
    print(f"Epoch {epoch.epoch}: Train Acc={epoch.train_accuracy:.2f}%, Val Acc={epoch.val_accuracy:.2f}%")
```

> **Note**: For detailed customization instructions, see the [Customizing for Your Project](#customizing-for-your-project) section.

## Troubleshooting

### Docker Issues

**Problem**: `Cannot connect to the Docker daemon`

**Solutions**:
1. **Start Docker Desktop** (macOS/Windows) or Docker daemon (Linux)
2. Verify Docker is running: `docker ps`
3. On macOS/Windows, open Docker Desktop application
4. Wait for Docker to fully start (check system tray icon)

**Problem**: `docker-compose: command not found`

**Solutions**:
1. On newer Docker installations, use: `docker compose` (without hyphen)
2. Or install Docker Compose separately: `pip install docker-compose`
3. Check Docker version: `docker --version` and `docker compose version`

### PostgreSQL Connection Issues

**Problem**: Cannot connect to PostgreSQL

**Solutions**:
1. **Verify container is running**:
   ```bash
   docker-compose ps
   # Should show "Up" status
   ```

2. **Check container logs**:
   ```bash
   docker-compose logs postgres
   # Look for errors or "database system is ready to accept connections"
   ```

3. **Wait for PostgreSQL to be ready**:
   ```bash
   # PostgreSQL takes 5-10 seconds to start
   sleep 10
   docker-compose ps
   # Health status should be "healthy"
   ```

4. **Verify port is not in use**:
   ```bash
   # macOS/Linux
   lsof -i :5432
   # Windows
   netstat -ano | findstr :5432
   ```

5. **Check `.env` file** (if created) has correct credentials matching `docker-compose.yml`

6. **Restart container**:
   ```bash
   docker-compose restart postgres
   ```

### Migration Errors

**Problem**: Alembic migration fails with connection error

**Solutions**:
1. **Ensure database is running**: `docker-compose up -d`
2. **Wait for PostgreSQL to be ready** (check with `docker-compose ps`)
3. **Verify connection**:
   ```bash
   docker exec mnist_postgres psql -U postgres -d mnist_db -c "SELECT 1;"
   ```

4. **Check database URL**: Ensure `.env` (if used) matches Docker Compose settings
5. **If tables already exist**, you may need to stamp:
   ```bash
   alembic stamp head
   ```

**Problem**: `Target database is not up to date`

**Solutions**:
```bash
# Check current migration state
alembic current

# Apply pending migrations
alembic upgrade head
```

### Dependency Installation Issues

**Problem**: `Failed to build wheel` or build errors

**Solutions**:
1. **Upgrade pip first**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Python 3.14+ compatibility**: Some packages may need newer versions. The `requirements.txt` uses `>=` to allow newer compatible versions.

3. **Install build dependencies** (if needed):
   ```bash
   # macOS
   xcode-select --install
   brew install postgresql

   # Ubuntu/Debian
   sudo apt-get install build-essential libpq-dev
   ```

4. **Install packages individually** to identify problematic dependency:
   ```bash
   pip install sqlalchemy
   pip install alembic
   pip install psycopg2-binary
   # etc.
   ```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solutions**:
1. **Ensure virtual environment is activated**:
   ```bash
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

2. **Verify you're in project root directory**:
   ```bash
   pwd  # Should show .../final-project-template
   ls src  # Should list source files
   ```

3. **Use module execution** (recommended):
   ```bash
   python -m src.main  # ✅ Correct
   # NOT: python src/main.py  # ❌ May fail
   ```

4. **Reinstall dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Training Issues

**Problem**: Training is very slow

**Solutions**:
1. Training time depends on your model and dataset - CPU training can be slow for large models
2. Reduce epochs in `.env`: `TRAINING_EPOCHS=2`
3. Reduce batch size: `TRAINING_BATCH_SIZE=32`
4. Use GPU if available: `TRAINING_DEVICE=cuda` (requires CUDA-enabled PyTorch)
5. Optimize your model architecture or use a smaller dataset for testing

**Problem**: Out of memory errors

**Solutions**:
1. Reduce batch size: `TRAINING_BATCH_SIZE=32` or `16`
2. Close other applications to free memory
3. Use smaller model architecture

### CUDA/GPU Issues

**Problem**: CUDA not available when using GPU

**Solutions**:
1. **Verify PyTorch CUDA installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   # Should print True if CUDA is available
   ```

2. **Set device to CPU** in `.env`:
   ```env
   TRAINING_DEVICE=cpu
   ```

3. **Install CUDA-enabled PyTorch** (if you have NVIDIA GPU):
   ```bash
   # Visit https://pytorch.org/get-started/locally/
   # Install appropriate version for your system
   ```

### Port Already in Use

**Problem**: Port 5432 already in use

**Solutions**:
1. **Find what's using the port**:
   ```bash
   # macOS/Linux
   lsof -i :5432
   # Windows
   netstat -ano | findstr :5432
   ```

2. **Stop conflicting service**:
   ```bash
   # macOS (if using Homebrew PostgreSQL)
   brew services stop postgresql

   # Linux
   sudo systemctl stop postgresql
   ```

3. **Use different port**:
   - Edit `docker-compose.yml`: Change `5432:5432` to `5433:5432`
   - Update `.env`: `POSTGRES_PORT=5433`
   - Update connection string accordingly

### Database Query Issues

**Problem**: `relation "training_runs" does not exist`

**Solutions**:
1. **Run migrations**:
   ```bash
   alembic upgrade head
   ```

2. **Verify tables exist**:
   ```bash
   docker exec mnist_postgres psql -U postgres -d mnist_db -c "\dt"
   ```

3. **Check you're using correct database**:
   ```bash
   docker exec mnist_postgres psql -U postgres -d mnist_db -c "SELECT current_database();"
   ```

### Still Having Issues?

1. **Check all services are running**:
   ```bash
   docker-compose ps
   python --version
   pip list | grep -E "(torch|sqlalchemy|alembic)"
   ```

2. **Review logs**:
   ```bash
   docker-compose logs postgres
   ```

3. **Start fresh** (if needed):
   ```bash
   # Stop and remove containers
   docker-compose down

   # Remove volumes (⚠️ deletes data)
   docker-compose down -v

   # Start fresh
   docker-compose up -d
   alembic upgrade head
   ```

## Quick Reference

### Common Commands

```bash
# Start PostgreSQL
docker-compose up -d

# Stop PostgreSQL
docker-compose down

# View logs
docker-compose logs postgres

# Run migrations
alembic upgrade head

# Train model
python -m src.main

# Check database
docker exec mnist_postgres psql -U postgres -d mnist_db -c "SELECT * FROM training_runs;"
```

### Project Workflow

1. **Start Docker** → `docker-compose up -d`
2. **Run Migrations** → `alembic upgrade head`
3. **Train Model** → `python -m src.main`
4. **Query Results** → Use SQL queries or GUI tool

### Alembic Commands

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history

# Show current revision
alembic current
```

## Customizing for Your Project

> **Quick Start**: If you're new to the project, complete the [Setup Instructions](#setup-instructions) first, then return here to customize for your specific model and dataset.

This boilerplate includes an MNIST example, but it's designed to be easily adapted for any model and dataset. Follow these steps to customize it for your specific use case.

### Step-by-Step Customization Guide

#### Step 1: Replace the Model Architecture

**File**: `src/models/mnist_model.py` (or create a new file)

```python
# Example: Custom CNN for CIFAR-10
import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 channels for RGB
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**Key considerations:**
- Input shape: Match your data (e.g., 3 channels for RGB images, 1 for grayscale)
- Output shape: Match number of classes or output dimensions
- Architecture: Design appropriate for your task (CNN, RNN, Transformer, etc.)

#### Step 2: Update the Trainer

**File**: `src/training/trainer.py`

**2a. Modify Data Loading** (`_get_data_loaders()` method):

```python
def _get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
    """Get training and validation data loaders."""
    # Example: CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 channels
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=self.config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=self.config.batch_size,
        shuffle=False
    )

    return train_loader, val_loader
```

**2b. Update Model Initialization**:

```python
def __init__(self, config: TrainingConfig):
    """Initialize trainer with configuration."""
    self.config = config
    self.device = torch.device(config.device)
    self.model = CIFAR10Net().to(self.device)  # Use your model
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
    self.training_history: List[Dict] = []
```

**2c. Customize Metrics** (if needed):

```python
def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
    """Validate the model."""
    self.model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # For regression tasks, you might calculate MAE/MSE instead
    # For multi-class, you might calculate F1, precision, recall

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy
```

#### Step 3: Update Main Script

**File**: `src/main.py`

```python
from src.models.cifar10_model import CIFAR10Net  # Import your model
# ... other imports ...

def main():
    db = get_database()
    session_gen = db.get_session()
    session = next(session_gen)

    try:
        training_service = create_training_service(session)

        training_run = training_service.train_and_save(
            model_name="CIFAR10Net",  # Update model name
            notes="Custom CIFAR-10 training"
        )
        # ... rest of code ...
```

#### Step 4: Update Service Layer (if trainer interface changes)

**File**: `src/services/training_service.py`

If your trainer returns different metrics, update the service:

```python
def train_and_save(self, model_name: str = "MyModel", notes: str = None) -> TrainingRun:
    training_results = self.trainer.train()

    training_run = TrainingRun(
        model_name=model_name,
        epochs=self.trainer.config.epochs,
        batch_size=self.trainer.config.batch_size,
        learning_rate=self.trainer.config.learning_rate,
        final_loss=training_results['final_loss'],
        final_accuracy=training_results.get('final_accuracy'),  # Use .get() if optional
        # Add any custom metrics here
        training_time_seconds=training_results['training_time_seconds'],
        notes=notes
    )
    # ... rest of code ...
```

#### Step 5: Update Configuration (Optional)

**File**: `src/config.py`

Add model-specific configuration:

```python
class TrainingConfig(BaseSettings):
    """Training configuration settings."""

    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 5
    device: str = "cpu"

    # NEW: Add model-specific config
    num_classes: int = 10
    image_size: int = 32
    use_augmentation: bool = True

    class Config:
        env_file = ".env"
        env_prefix = "TRAINING_"
        case_sensitive = False
```

#### Step 6: Update Database Schema (if needed)

See the [Customizing Database Schema](#customizing-database-schema-and-alembic-migrations) section for detailed instructions on:
- Adding new columns for custom metrics
- Creating Alembic migrations
- Updating repositories and services

### Example: Complete Customization for Regression Task

**1. Model** (`src/models/regression_model.py`):

```python
class RegressionNet(nn.Module):
    def __init__(self, input_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Single output for regression

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

**2. Trainer** - Update metrics calculation:

```python
def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
    """Calculate MAE instead of accuracy."""
    self.model.eval()
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            mae = torch.mean(torch.abs(outputs - targets))
            total_mae += mae.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_mae = total_mae / total_samples
    return avg_mae, avg_mae  # Return MAE for both loss and "accuracy"
```

**3. Database Models** - Add regression metrics (see Alembic section)

**4. Service** - Update to save regression metrics

### Quick Reference: What to Change

| Component | File | What to Modify |
|-----------|------|----------------|
| Model Architecture | `src/models/*.py` | Define your PyTorch model |
| Data Loading | `src/training/trainer.py` | `_get_data_loaders()` method |
| Metrics | `src/training/trainer.py` | `_validate()` and `_train_epoch()` |
| Model Instance | `src/training/trainer.py` | `__init__()` - model initialization |
| Main Entry | `src/main.py` | Import and model name |
| Database Schema | `src/database/models.py` | Add columns for custom metrics |
| Migrations | `migrations/versions/` | Generate with Alembic |

### Testing Your Customization

After making changes:

1. **Test imports**:
   ```bash
   python -c "from src.models.your_model import YourModel; print('✓ Model OK')"
   ```

2. **Test data loading**:
   ```bash
   python -c "from src.training.trainer import YourTrainer; t = YourTrainer(config); print('✓ Trainer OK')"
   ```

3. **Run training**:
   ```bash
   python -m src.main
   ```

4. **Verify database**:
   ```bash
   docker exec mnist_postgres psql -U postgres -d mnist_db -c "SELECT * FROM training_runs;"
   ```

## Customizing Database Schema and Alembic Migrations

When adapting this boilerplate for your specific model, you may need to track different metrics or add additional columns. This section explains how to modify the database schema and create Alembic migrations.

### Understanding the Current Schema

The boilerplate includes two main tables:

1. **`training_runs`**: Stores metadata about each training run
   - Current fields: `id`, `model_name`, `epochs`, `batch_size`, `learning_rate`, `final_loss`, `final_accuracy`, `training_time_seconds`, `created_at`, `notes`

2. **`epoch_results`**: Stores per-epoch metrics
   - Current fields: `id`, `training_run_id`, `epoch`, `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `created_at`

### Step 1: Modify Database Models

Edit `src/database/models.py` to add or modify columns:

**Example: Adding F1 Score and Precision/Recall**

```python
from sqlalchemy import Column, Integer, Float, String, DateTime, Text
from sqlalchemy.sql import func
from src.database.base import Base


class TrainingRun(Base):
    """Model for storing training run metadata."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)

    # Existing metrics
    final_loss = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)

    # NEW: Add your custom metrics
    final_f1_score = Column(Float, nullable=True)  # For classification tasks
    final_precision = Column(Float, nullable=True)
    final_recall = Column(Float, nullable=True)

    # NEW: Add model-specific metadata
    model_parameters = Column(Integer, nullable=True)  # Number of parameters
    optimizer_name = Column(String(50), nullable=True)  # e.g., 'adam', 'sgd'

    training_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)


class EpochResult(Base):
    """Model for storing per-epoch training results."""
    __tablename__ = "epoch_results"

    id = Column(Integer, primary_key=True, index=True)
    training_run_id = Column(Integer, nullable=False, index=True)
    epoch = Column(Integer, nullable=False)

    # Existing metrics
    train_loss = Column(Float, nullable=False)
    train_accuracy = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)

    # NEW: Add per-epoch custom metrics
    train_f1_score = Column(Float, nullable=True)
    val_f1_score = Column(Float, nullable=True)
    learning_rate_used = Column(Float, nullable=True)  # Track LR if using scheduler

    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

**Example: For Regression Tasks**

```python
class TrainingRun(Base):
    # ... existing fields ...

    # Replace accuracy metrics with regression metrics
    final_mae = Column(Float, nullable=True)  # Mean Absolute Error
    final_mse = Column(Float, nullable=True)  # Mean Squared Error
    final_rmse = Column(Float, nullable=True)  # Root Mean Squared Error
    final_r2_score = Column(Float, nullable=True)  # R² Score


class EpochResult(Base):
    # ... existing fields ...

    # Replace accuracy with regression metrics
    train_mae = Column(Float, nullable=True)
    val_mae = Column(Float, nullable=True)
    train_mse = Column(Float, nullable=True)
    val_mse = Column(Float, nullable=True)
```

### Step 2: Generate Alembic Migration

After modifying the models, create a new migration:

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Generate migration (Alembic will detect changes automatically)
alembic revision --autogenerate -m "add f1_score and precision_recall metrics"
```

**What Alembic does:**
- Compares your current models with the database schema
- Generates a migration file in `migrations/versions/`
- Includes `upgrade()` and `downgrade()` functions

### Step 3: Review the Generated Migration

Open the generated migration file (e.g., `migrations/versions/002_add_f1_score.py`):

```python
"""add f1_score and precision_recall metrics

Revision ID: 002_add_f1_score
Revises: 001_initial
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002_add_f1_score'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to training_runs
    op.add_column('training_runs', sa.Column('final_f1_score', sa.Float(), nullable=True))
    op.add_column('training_runs', sa.Column('final_precision', sa.Float(), nullable=True))
    op.add_column('training_runs', sa.Column('final_recall', sa.Float(), nullable=True))

    # Add new columns to epoch_results
    op.add_column('epoch_results', sa.Column('train_f1_score', sa.Float(), nullable=True))
    op.add_column('epoch_results', sa.Column('val_f1_score', sa.Float(), nullable=True))


def downgrade() -> None:
    # Remove columns (reverse order)
    op.drop_column('epoch_results', 'val_f1_score')
    op.drop_column('epoch_results', 'train_f1_score')
    op.drop_column('training_runs', 'final_recall')
    op.drop_column('training_runs', 'final_precision')
    op.drop_column('training_runs', 'final_f1_score')
```

**Important:** Always review the generated migration before applying it!

### Step 4: Apply the Migration

```bash
# Apply the migration
alembic upgrade head
```

**Verify the changes:**
```bash
# Check the new columns exist
docker exec mnist_postgres psql -U postgres -d mnist_db -c "\d training_runs"
docker exec mnist_postgres psql -U postgres -d mnist_db -c "\d epoch_results"
```

### Step 5: Update Your Code to Use New Columns

**Update Repository** (`src/database/repository.py`):

```python
# No changes needed - repositories work with SQLAlchemy models automatically
# Just ensure you're creating TrainingRun/EpochResult objects with new fields
```

**Update Trainer** (`src/training/trainer.py`):

```python
def train(self) -> Dict:
    # ... existing training code ...

    # Calculate additional metrics
    final_f1 = calculate_f1_score(...)  # Your metric calculation
    final_precision = calculate_precision(...)
    final_recall = calculate_recall(...)

    final_result = {
        'final_loss': self.training_history[-1]['val_loss'],
        'final_accuracy': self.training_history[-1]['val_accuracy'],
        'final_f1_score': final_f1,  # NEW
        'final_precision': final_precision,  # NEW
        'final_recall': final_recall,  # NEW
        'training_time_seconds': training_time,
        'epoch_results': self.training_history
    }
    return final_result
```

**Update Service** (`src/services/training_service.py`):

```python
def train_and_save(self, model_name: str = "MyModel", notes: str = None) -> TrainingRun:
    training_results = self.trainer.train()

    training_run = TrainingRun(
        model_name=model_name,
        epochs=self.trainer.config.epochs,
        batch_size=self.trainer.config.batch_size,
        learning_rate=self.trainer.config.learning_rate,
        final_loss=training_results['final_loss'],
        final_accuracy=training_results['final_accuracy'],
        final_f1_score=training_results['final_f1_score'],  # NEW
        final_precision=training_results['final_precision'],  # NEW
        final_recall=training_results['final_recall'],  # NEW
        training_time_seconds=training_results['training_time_seconds'],
        notes=notes
    )

    # ... rest of the code ...
```

### Common Migration Scenarios

#### Scenario 1: Adding a New Column

```python
# In models.py
new_field = Column(String(100), nullable=True)

# Generate migration
alembic revision --autogenerate -m "add new_field"
alembic upgrade head
```

#### Scenario 2: Changing Column Type

```python
# In models.py - change from String to Text
notes = Column(Text, nullable=True)  # Was String(500)

# Generate migration
alembic revision --autogenerate -m "change notes to text"
# Review migration - may need manual adjustment for data conversion
alembic upgrade head
```

#### Scenario 3: Adding Index

```python
# In models.py
from sqlalchemy import Index

# Add index on model_name for faster queries
Index('idx_model_name', TrainingRun.model_name)

# Generate migration
alembic revision --autogenerate -m "add index on model_name"
alembic upgrade head
```

#### Scenario 4: Removing a Column

```python
# In models.py - remove the column definition

# Generate migration
alembic revision --autogenerate -m "remove unused_column"
# ⚠️ WARNING: This will delete data! Review migration carefully.
alembic upgrade head
```

### Manual Migration Editing

Sometimes you need to edit migrations manually:

**Example: Adding a default value to existing rows**

```python
def upgrade() -> None:
    # Add column
    op.add_column('training_runs', sa.Column('optimizer_name', sa.String(50), nullable=True))

    # Set default value for existing rows
    op.execute("UPDATE training_runs SET optimizer_name = 'adam' WHERE optimizer_name IS NULL")

    # Now you can make it NOT NULL if desired
    op.alter_column('training_runs', 'optimizer_name', nullable=False)
```

### Migration Best Practices

1. **Always review generated migrations** before applying
2. **Test migrations on a copy** of production data first
3. **Use descriptive migration messages**: `alembic revision -m "descriptive message"`
4. **Keep migrations small** - one logical change per migration
5. **Never edit applied migrations** - create a new one instead
6. **Backup database** before running migrations in production

### Rolling Back Migrations

If something goes wrong:

```bash
# Rollback one migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade 001_initial

# Rollback all migrations
alembic downgrade base
```

### Complete Example: Customizing for Image Classification

Here's a complete example of customizing for a different use case:

**1. Update Models** (`src/database/models.py`):

```python
class TrainingRun(Base):
    # ... existing fields ...

    # Image classification specific
    num_classes = Column(Integer, nullable=False)
    image_size = Column(String(20), nullable=True)  # e.g., "224x224"
    augmentation_used = Column(String(200), nullable=True)  # JSON string
    pretrained_model = Column(String(100), nullable=True)  # e.g., "resnet50"

    # Per-class accuracy
    top1_accuracy = Column(Float, nullable=True)
    top5_accuracy = Column(Float, nullable=True)  # For ImageNet-style tasks
```

**2. Generate Migration**:

```bash
alembic revision --autogenerate -m "add image classification fields"
```

**3. Apply Migration**:

```bash
alembic upgrade head
```

**4. Update Trainer** to populate new fields:

```python
training_run = TrainingRun(
    # ... existing fields ...
    num_classes=10,
    image_size="28x28",
    top1_accuracy=final_top1_acc,
    top5_accuracy=final_top5_acc,
)
```

### Troubleshooting Migrations

**Problem**: `Target database is not up to date`

```bash
# Check current revision
alembic current

# See migration history
alembic history

# Apply pending migrations
alembic upgrade head
```

**Problem**: Migration conflicts

```bash
# If migrations are out of sync, you may need to stamp
alembic stamp head  # Mark current state as up-to-date
```

**Problem**: Need to modify existing migration before applying

```bash
# Edit the migration file in migrations/versions/
# Then apply it
alembic upgrade head
```

## Next Steps

After setting up this boilerplate, consider:

1. **Add Testing**: Write unit tests for each module using pytest
2. **Add Logging**: Implement structured logging with Python's logging module
3. **Add API**: Create a REST API using FastAPI or Flask to query training results
4. **Add Visualization**: Create dashboards with matplotlib/plotly for training metrics
5. **Add Model Serving**: Deploy model for inference using Flask/FastAPI
6. **Add CI/CD**: Set up GitHub Actions for automated testing and deployment
7. **Experiment**: Try different model architectures, hyperparameters, or datasets
8. **Add Monitoring**: Track training metrics over time and compare runs

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Datasets](https://pytorch.org/vision/stable/datasets.html)

## Testing the Setup

To verify everything is working correctly:

```bash
# 1. Check Docker is running
docker ps

# 2. Check PostgreSQL container
docker-compose ps

# 3. Test database connection
docker exec mnist_postgres psql -U postgres -d mnist_db -c "SELECT version();"

# 4. Test Python imports
python -c "from src.config import get_database_config; print('✓ Config OK')"
python -c "from src.database.base import get_database; print('✓ Database OK')"
python -c "from src.models.mnist_model import MNISTNet; print('✓ Model OK')"

# 5. Run a quick training test
python -m src.main
```

If all steps complete without errors, your setup is correct! 🎉

## FAQ

**Q: Do I need to create a `.env` file?**
A: No, the application uses sensible defaults. Create `.env` only if you want to customize settings.

**Q: Can I use this for models other than MNIST?**
A: Absolutely! This is a boilerplate template. Modify the model architecture in `src/models/`, update the trainer in `src/training/trainer.py` for your dataset, and adjust `src/main.py` as needed.

**Q: How long does training take?**
A: Depends on your model and dataset. The MNIST example takes ~2-5 minutes for 5 epochs on CPU. Your model may vary.

**Q: Can I use GPU for training?**
A: Yes, set `TRAINING_DEVICE=cuda` in `.env`, but you'll need CUDA-enabled PyTorch installed.

**Q: Where is the dataset stored?**
A: In the `./data/` directory, automatically created on first run. You can modify this location in the trainer.

**Q: How do I reset the database?**
A: Run `docker-compose down -v` to remove containers and volumes, then start fresh.

**Q: Can I run multiple training runs?**
A: Yes! Each run creates a new entry in `training_runs` table with a unique ID.

**Q: How do I change training parameters?**
A: Create a `.env` file and set `TRAINING_EPOCHS`, `TRAINING_BATCH_SIZE`, etc.

**Q: How do I adapt this for my own model/dataset?**
A: See the comprehensive [Customizing for Your Project](#customizing-for-your-project) section, which includes step-by-step instructions with examples for different use cases (classification, regression, etc.).

**Q: How do I add custom metrics to track?**
A: See the [Customizing Database Schema and Alembic Migrations](#customizing-database-schema-and-alembic-migrations) section for detailed instructions on modifying the database schema and creating migrations.

**Q: How do I set up code formatting and linting?**
A: Install development dependencies (`pip install -r requirements-dev.txt`) and run `pre-commit install`. This sets up ruff for automatic code formatting and linting on every commit.

## Code Quality and Formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [pre-commit](https://pre-commit.com/) hooks to ensure code quality.

### Setting Up Pre-commit Hooks

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

After setup, pre-commit will automatically:
- Format code using ruff
- Sort imports
- Check for common issues (trailing whitespace, merge conflicts, etc.)

### Running Manually

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run ruff only
ruff check .
ruff format .

# Auto-fix issues
ruff check --fix .
```

### Pre-commit Configuration

The project includes:
- **Ruff linter**: Catches errors and enforces code style
- **Ruff formatter**: Formats code consistently
- **Import sorting**: Automatically sorts imports
- **Basic checks**: Trailing whitespace, file endings, etc.

Configuration files:
- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `pyproject.toml`: Ruff linting and formatting rules

## CI/CD Pipeline

This project includes a GitHub Actions CI/CD pipeline that automatically:

### What the Pipeline Does

1. **Lint Check** (on every push/PR):
   - Runs ruff linter to check code quality
   - Verifies code formatting with ruff formatter
   - Ensures code follows project standards

2. **Training Test** (on every push/PR):
   - Runs training with 2 epochs (configurable via environment variables)
   - Displays training results in the workflow logs
   - Verifies the training pipeline works correctly
   - **Note**: Training runs without database persistence in CI

### Workflow File

The CI/CD configuration is in `.github/workflows/ci.yml`. It runs:
- On pushes to `main`, `master`, or `develop` branches
- On pull requests to those branches

### Viewing CI Results

1. Go to your GitHub repository
2. Click on the "Actions" tab
3. View workflow runs and their results
4. Click on a run to see detailed logs, including:
   - Linting results
   - Training progress and final metrics
   - Any errors or warnings

### CI Training Configuration

The CI training uses environment variables (set in the workflow):
- `TRAINING_EPOCHS=2` (reduced for faster CI runs)
- `TRAINING_BATCH_SIZE=64`
- `TRAINING_DEVICE=cpu`

You can modify these in `.github/workflows/ci.yml` if needed.

### Running CI Training Locally

To test the CI training script locally:

```bash
# Set environment variables (optional)
export TRAINING_EPOCHS=2
export TRAINING_BATCH_SIZE=64

# Run CI training script
python -m src.ci_train
```

This will run training without database persistence, just like in CI.

## License

This is a boilerplate template project for educational purposes. Feel free to use, modify, and adapt it for your own projects.

---

## Summary

This boilerplate provides:

✅ **Complete ML Training Infrastructure**: Docker Compose setup, PostgreSQL database, Alembic migrations
✅ **Clean Architecture**: Follows SRP and Dependency Injection principles
✅ **Flexible Design**: Easy to adapt for any model and dataset
✅ **Production-Ready Patterns**: Repository pattern, service layer, configuration management
✅ **Comprehensive Documentation**: Step-by-step guides for setup, customization, and troubleshooting

**Getting Started**: Follow the [Setup Instructions](#setup-instructions) to get started, then customize for your specific use case using the [Customizing for Your Project](#customizing-for-your-project) guide.

**Need Help?**: Check the [Troubleshooting](#troubleshooting) section or [FAQ](#faq) for common issues and solutions.
