# =============================================================================
# JUPYTER NOTEBOOK CONFIG — NEXUS Studies Environment
# =============================================================================
# Optimized for: Colab local runtime connection, large outputs, GPU monitoring
# =============================================================================

c = get_config()  # noqa: F821

# -----------------------------------------------------------------------------
# Server & Security
# -----------------------------------------------------------------------------
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_origin = "https://colab.research.google.com"
c.ServerApp.allow_remote_access = True
c.ServerApp.root_dir = "/home/jovyan/work"
c.ServerApp.preferred_dir = "/home/jovyan/work/notebooks"

# Token auth (set via env JUPYTER_TOKEN)
c.ServerApp.token = "nexus-study"
c.ServerApp.password = ""  # Disabled, token only

# -----------------------------------------------------------------------------
# Resource Limits (prevent OOM on large outputs)
# -----------------------------------------------------------------------------
c.ServerApp.max_body_size = 500 * 1024 * 1024  # 500MB
c.ServerApp.max_buffer_size = 512 * 1024 * 1024  # 512MB

# -----------------------------------------------------------------------------
# Extensions
# -----------------------------------------------------------------------------
c.ServerApp.jpserver_extensions = {
    "jupyter_server_proxy": True,
    "jupyterlab_git": True,
}

# -----------------------------------------------------------------------------
# Notebook Settings
# -----------------------------------------------------------------------------
c.NotebookApp.notebook_dir = "/home/jovyan/work/notebooks"
c.NotebookApp.enable_mathjax = True

# -----------------------------------------------------------------------------
# Kernel Management
# -----------------------------------------------------------------------------
c.MappingKernelManager.cull_idle_timeout = 3600  # 1 hour
c.MappingKernelManager.cull_interval = 300  # 5 min
c.MappingKernelManager.cull_connected = False

# -----------------------------------------------------------------------------
# Contents Manager (large files, git integration)
# -----------------------------------------------------------------------------
c.FileContentsManager.max_file_size = 100 * 1024 * 1024  # 100MB
c.FileContentsManager.always_delete_dir = False

# -----------------------------------------------------------------------------
# Proxy (for MCP, Ollama, TensorBoard, etc)
# -----------------------------------------------------------------------------
c.ServerProxy.servers = {
    "mcp-proxy": {
        "command": ["socat", "TCP-LISTEN:7475,fork,reuseaddr", "TCP:host.docker.internal:7474"],
        "port": 7475,
        "absolute_url": True,
        "timeout": 30,
        "launcher_entry": {
            "enabled": True,
            "title": "NEXUS MCP Proxy",
        },
    },
    "ollama": {
        "command": ["socat", "TCP-LISTEN:11434,fork,reuseaddr", "TCP:ollama:11434"],
        "port": 11434,
        "absolute_url": True,
        "launcher_entry": {"enabled": True, "title": "Ollama API"},
    },
    "tensorboard": {
        "command": ["tensorboard", "--logdir", "/home/jovyan/work/logs", "--host", "0.0.0.0", "--port", "6006"],
        "port": 6006,
        "absolute_url": True,
        "launcher_entry": {"enabled": True, "title": "TensorBoard"},
    },
}

# -----------------------------------------------------------------------------
# Terminal Settings
# -----------------------------------------------------------------------------
c.TerminalInteractiveShell.term_title = True
c.TerminalInteractiveShell.autoawait = True