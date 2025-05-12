# config.py - Shared configuration parameters

# Model parameters
BATCH_SIZE = 128
EPOCHS = 5
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# MNIST dataset statistics
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Database configuration
DB_CONFIG = {
    "host_env": "DB_HOST",
    "host_default": "localhost",
    "db_env": "DB_NAME",
    "db_default": "mnist_logs",
    "user_env": "DB_USER",
    "user_default": "mnist_user",
    "pass_env": "DB_PASS",
    "pass_default": "mnist_pass"
} 