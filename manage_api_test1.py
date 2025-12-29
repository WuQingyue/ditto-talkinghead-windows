import subprocess
import os
import sys
import time
import signal
import atexit
import yaml # 需要 pip install PyYAML
from pathlib import Path
from typing import List, Optional
import mimetypes
import socket
import shutil

from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# # --- Constants ---
# SCRIPT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

# --- Determine base directory ---
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running in a PyInstaller bundle
    # sys._MEIPASS is the path to the temporary folder where PyInstaller unpacks data
    SCRIPT_DIR = Path(sys._MEIPASS)
    print(f"INFO: Running in PyInstaller bundle. MEIPASS: {SCRIPT_DIR}")
else:
    # Running as a normal Python script
    SCRIPT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
    print(f"INFO: Running as script. SCRIPT_DIR: {SCRIPT_DIR}")

# --- Determine config file based on OS ---
DEFAULT_CONFIG_FILE_NAME = "manager_config.yaml"
LINUX_CONFIG_FILE_NAME = "manager_config_linux.yaml"
CONFIG_FILE_TO_USE = DEFAULT_CONFIG_FILE_NAME

if sys.platform.startswith("linux"):
    linux_config_path = SCRIPT_DIR / LINUX_CONFIG_FILE_NAME
    if linux_config_path.exists():
        CONFIG_FILE_TO_USE = LINUX_CONFIG_FILE_NAME
        print(f"INFO: Running on Linux, using Linux-specific config: {linux_config_path}")
    else:
        print(f"INFO: Running on Linux, Linux-specific config '{LINUX_CONFIG_FILE_NAME}' not found at {linux_config_path}. Using default: {DEFAULT_CONFIG_FILE_NAME}")
else:
    print(f"INFO: Not on Linux (platform: {sys.platform}), using default config: {DEFAULT_CONFIG_FILE_NAME}")

CONFIG_FILE = SCRIPT_DIR / CONFIG_FILE_TO_USE

# --- 新增: 角色圖片服務的配置 ---
# BASE_CHARACTERS_DIR = SCRIPT_DIR / "livetalking-onnx" / "data" / "avatars"
# PREVIEW_IMAGE_FILENAME = "00000001.png"
# PREVIEW_IMAGE_SUBFOLDER = "full_imgs"

BASE_CHARACTERS_DIR = SCRIPT_DIR / "ditto-talkinghead-windows" / "data" / "avatars"
PREVIEW_IMAGE_FILENAME = "00000001.png"
PREVIEW_IMAGE_SUBFOLDER = "full_imgs"

# --- Global state ---
config = {}
server_processes = {} # Dictionary to store Popen objects: {'name': popen_obj}

# --- Helper Functions ---
def get_python_executable(resolved_env_path_str: str) -> str:
    """Gets the path to the python executable within the resolved env path."""
    resolved_env_path = Path(resolved_env_path_str)
    if sys.platform == "win32" or sys.platform == "cygwin":
        return str(resolved_env_path / "python.exe")
    else: # Linux, macOS, etc.
        return str(resolved_env_path / "bin" / "python")

def resolve_path(relative_path_str: str) -> str:
    """Resolves a path relative to the manager script's directory."""
    relative_path = Path(relative_path_str)
    if relative_path.is_absolute():
        return str(relative_path)
    return str((SCRIPT_DIR / relative_path).resolve())

# --- FastAPI Setup ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global config
    print(f"Manager script directory: {SCRIPT_DIR}")
    print(f"Loading configuration from: {CONFIG_FILE}...")
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config or 'environments' not in config:
             print("ERROR: Config file is empty, invalid, or missing 'environments' key.")
             raise RuntimeError("Failed to load valid configuration.")
        print("Configuration loaded successfully.")

        if not BASE_CHARACTERS_DIR.exists():
            print(f"Warning: Character base directory {BASE_CHARACTERS_DIR} does not exist. API for characters might return empty results.")

    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {CONFIG_FILE}")
        raise RuntimeError("Configuration file missing.")
    except Exception as e:
        print(f"ERROR: Failed to load or parse config file: {e}")
        raise RuntimeError("Failed to load configuration.")

    yield

    print("Manager shutting down. Stopping all managed servers...")
    stop_servers()
    print("Cleanup complete.")

app = FastAPI(title="Multi-Server Manager & Image API", lifespan=lifespan)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
# class LiveTalkingParams(BaseModel):
#     avatar_id: str | None = None
#     listen_port: str | None = None
#     fastapi_tts_port: str | None = None

class DittoParams(BaseModel):
    avatar_id: str | None = None
    listen_port: str | None = None
    fastapi_tts_port: str | None = None


class CharacterInfo(BaseModel):
    id: str
    name: str
    preview_image_url: Optional[str] = None

# --- Core Logic Functions ---
def stop_single_process(name, process):
    if process.poll() is None:
        print(f"Stopping server '{name}' (PID: {process.pid})...")
        try:
            process.terminate()
            process.wait(timeout=10)
            return f"Server '{name}' terminated gracefully."
        except subprocess.TimeoutExpired:
            print(f"Graceful shutdown timed out for '{name}'. Forcing kill...")
            try:
                process.kill()
                process.wait(timeout=5)
                return f"Server '{name}' killed after timeout."
            except Exception as kill_e:
                return f"Failed to force kill '{name}': {kill_e}"
        except ProcessLookupError:
            return f"Process for '{name}' not found (already stopped?)."
        except Exception as e:
            try: process.kill()
            except: pass
            return f"Error stopping server '{name}': {e}."
    else:
        return f"Server '{name}' was already stopped."

def stop_servers():
    global server_processes
    if not server_processes: return {}
    print("Stopping all tracked server processes...")
    results = {}
    for name, process in list(server_processes.items()):
        message = stop_single_process(name, process)
        results[name] = message
        if name in server_processes:
            del server_processes[name]
    print("Server stop process complete.")
    return results

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
        except socket.error as e:
            if e.errno == socket.errno.EADDRINUSE:
                print(f"Port {port} is already in use.")
                return True
            else:
                print(f"Error checking port {port}: {e}")
                return True
        return False

# --- API Endpoints ---
@app.post("/start/{server_name}", summary="Start a specific server")
# async def start_single_server(server_name: str, params: LiveTalkingParams | None = None):
async def start_single_server(server_name: str, params: DittoParams | None = None):
    global server_processes, config
    if server_name not in config.get('environments', {}):
        raise HTTPException(status_code=404, detail=f"Server configuration '{server_name}' not found.")

    if server_name in server_processes and server_processes[server_name].poll() is None:
        return {"message": f"Server '{server_name}' is already running.", "pid": server_processes[server_name].pid}

    server_config = config['environments'][server_name]
    try:
        # env_path is primary, project_root for CWD, script relative to CWD
        # No direct use of server_config['env_path'] here for python_exe_to_use initialization
        working_dir_abs_str = resolve_path(server_config['project_root'])
        script_rel = server_config['script']
    except KeyError as e:
         raise HTTPException(status_code=500, detail=f"Missing required config key '{e}' for server '{server_name}'")

    python_exe_to_use = None

    # --- Python Executable Discovery ---
    # The primary method for all OS will be to use 'env_path' from the config.
    # get_python_executable handles the OS-specifics (e.g., /bin/python vs python.exe).
    if 'env_path' not in server_config:
        raise HTTPException(status_code=500, detail=f"Configuration error: 'env_path' missing for server '{server_name}'.")

    resolved_env_path_str = resolve_path(server_config['env_path'])
    python_exe_to_use = get_python_executable(resolved_env_path_str)

    if not Path(python_exe_to_use).exists():
        error_detail = (f"Python executable not found at resolved path: {python_exe_to_use} "
                        f"(derived from env_path: '{server_config['env_path']}' which resolved to '{resolved_env_path_str}') "
                        f"for server '{server_name}'.")
        raise HTTPException(status_code=500, detail=error_detail)
    
    print(f"INFO: Using Python executable: {python_exe_to_use} (derived from env_path: '{server_config['env_path']}') for server '{server_name}'")

    # Original script path (relative to working_dir) and base arguments
    base_args = server_config.get('command_args', [])
    final_args = [script_rel] + base_args

    # Determine port and check if it's in use
    port_to_check = None
    port_arg_name = None
    service_specific_port_value = None

    # if server_name == "livetalking_onnx":
    #     defaults = config.get('defaults', {}).get('livetalking_onnx', {})
    #     service_specific_port_value = (params.listen_port if params and params.listen_port is not None else None) or defaults.get('listen_port', '8001')
    #     port_arg_name = "--listenport"
    #     if "--avatar_id" not in base_args: final_args.extend(["--avatar_id", str((params.avatar_id if params and params.avatar_id is not None else None) or defaults.get('avatar_id', 'default_avatar'))])
    #     if port_arg_name not in base_args: final_args.extend([port_arg_name, str(service_specific_port_value)])

    if server_name == "ditto":
        defaults = config.get('defaults', {}).get('ditto', {})
        service_specific_port_value = (params.listen_port if params and params.listen_port is not None else None) or defaults.get('listen_port', '8010')
        port_arg_name = "--listenport"
        # --transport webrtc 已在 command_args 中配置，无需再次添加
        # 添加 --avatar_id 参数（如果不在 base_args 中）
        if "--avatar_id" not in base_args: final_args.extend(["--avatar_id", str((params.avatar_id if params and params.avatar_id is not None else None) or defaults.get('avatar_id', 'default_avatar'))])
        # 添加 --listenport 参数（如果不在 base_args 中，根据配置注释，此参数由 API manager 动态添加）
        if port_arg_name not in base_args: final_args.extend([port_arg_name, str(service_specific_port_value)])

    elif server_name == "fastapi_tts":
        defaults = config.get('defaults', {}).get('fastapi_tts', {})
        service_specific_port_value = (params.fastapi_tts_port if params and params.fastapi_tts_port is not None else None) or defaults.get('port', '50001')
        port_arg_name = "--port"
        if port_arg_name not in base_args:
            final_args.extend([port_arg_name, str(service_specific_port_value)])

    if service_specific_port_value:
        try:
            port_to_check = int(service_specific_port_value)
            if is_port_in_use(port_to_check):
                raise HTTPException(status_code=409, detail=f"Port {port_to_check} for server '{server_name}' is already in use.")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid port number '{service_specific_port_value}' for server '{server_name}'.")

    full_command = [python_exe_to_use] + final_args

    print(f"Attempting to start server '{server_name}'...")
    print(f"  Command: {' '.join(map(str,full_command))}")
    print(f"  CWD: {working_dir_abs_str}")

    try:
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_CONSOLE

        process = subprocess.Popen(
            full_command,
            cwd=working_dir_abs_str,
            creationflags=creation_flags,
        )
        server_processes[server_name] = process
        time.sleep(1.5)

        if process.poll() is not None:
            error_detail = f"Server '{server_name}' failed to start or terminated immediately (return code: {process.returncode}). Check its console/logs."
            if server_name in server_processes: del server_processes[server_name]
            raise HTTPException(status_code=500, detail=error_detail)

        return {"message": f"Server '{server_name}' started successfully.", "pid": process.pid}
    except Exception as e:
         if server_name in server_processes: del server_processes[server_name]
         raise HTTPException(status_code=500, detail=f"Failed to start server '{server_name}': {type(e).__name__}: {str(e)}")


@app.post("/start_all", summary="Start all configured servers")
# async def start_all_servers_api(params: LiveTalkingParams | None = None):
async def start_all_servers_api(params: DittoParams | None = None):
    results = {}
    has_errors = False # Keep for potential future use or more detailed error reporting
    start_attempted = False
    for server_name in config.get('environments', {}):
        start_attempted = True
        try:
            # current_params_for_server = LiveTalkingParams()
            current_params_for_server = DittoParams()
            if params:
                # if server_name == 'livetalking_onnx':
                #     current_params_for_server.listen_port = params.listen_port
                #     current_params_for_server.avatar_id = params.avatar_id
                if server_name == 'ditto':
                    current_params_for_server.listen_port = params.listen_port
                    current_params_for_server.avatar_id = params.avatar_id
                elif server_name == 'fastapi_tts':
                    current_params_for_server.fastapi_tts_port = params.fastapi_tts_port
            
            result = await start_single_server(server_name, params=current_params_for_server)
            results[server_name] = result
        except HTTPException as e:
            results[server_name] = {"error": e.detail, "status_code": e.status_code}
            has_errors = True
        except Exception as e:
            results[server_name] = {"error": f"Unexpected error for {server_name}: {type(e).__name__}: {str(e)}", "status_code": 500}
            has_errors = True
    if not start_attempted:
        return {"message": "No servers found in configuration."}
    return results


@app.post("/stop/{server_name}", summary="Stop a specific server")
async def stop_single_server_api(server_name: str):
    global server_processes
    if server_name not in config.get('environments', {}):
         raise HTTPException(status_code=404, detail=f"Server configuration '{server_name}' not found.")
    if server_name not in server_processes:
        return {"message": f"Server '{server_name}' is not currently running or tracked."}

    process = server_processes[server_name]
    message = stop_single_process(server_name, process)
    if server_name in server_processes:
        del server_processes[server_name]
    return {"message": message}

@app.post("/stop_all", summary="Stop all managed servers")
async def stop_all_servers_api():
    results = stop_servers()

    # 停止所有服务后，清理项目根目录下 uploads 目录中的所有文件和子目录
    try:
        uploads_dir = (SCRIPT_DIR / "uploads").resolve()
        if uploads_dir.exists() and uploads_dir.is_dir():
            for entry in uploads_dir.iterdir():
                try:
                    if entry.is_file() or entry.is_symlink():
                        entry.unlink(missing_ok=True) if hasattr(entry, 'unlink') else os.remove(str(entry))
                    elif entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                except Exception:
                    # 保持 API 可用性，记录异常由外层日志处理
                    pass
    except Exception:
        pass

    if not results:
        return {"message": "No active servers were found to stop, or no servers tracked."}
    return results

@app.get("/status", summary="Get status of all managed servers")
async def get_server_status():
    global server_processes, config
    status_report = {}
    configured_servers = config.get('environments', {}).keys()

    for server_name in configured_servers:
        if server_name in server_processes:
            process = server_processes[server_name]
            if process.poll() is None:
                status_report[server_name] = {"status": "running", "pid": process.pid}
            else:
                status_report[server_name] = {"status": "stopped", "pid": process.pid, "return_code": process.returncode}
        else:
            status_report[server_name] = {"status": "stopped", "pid": None}

    for name in list(server_processes.keys()):
        process = server_processes[name]
        if process.poll() is not None:
            print(f"Cleaning up tracker for detected stopped process: {name}")
            del server_processes[name]
            if name in configured_servers and (name not in status_report or status_report[name].get("status") == "running"):
                 status_report[name] = {"status": "stopped", "pid": process.pid, "return_code": process.returncode}
    return status_report

@app.get("/avatars", summary="Get available avatar IDs")
async def get_avatar_ids():
    try:
        # livetalking_root = config.get('environments', {}).get('livetalking_onnx', {}).get('project_root')
        # if not livetalking_root:
        #     avatars_path_str = resolve_path("./livetalking-onnx/data/avatars")
        # else:
        #     avatars_path_str = str(Path(resolve_path(livetalking_root)) / "data" / "avatars")
        ditto_root = config.get('environments', {}).get('ditto', {}).get('project_root')
        if not ditto_root:
            avatars_path_str = resolve_path("./ditto_data/avatars")
        else:
            avatars_path_str = str(Path(resolve_path(ditto_root)) / "data" / "avatars")
        avatars_path = Path(avatars_path_str)
        if not avatars_path.exists() or not avatars_path.is_dir():
            return {"message": f"Avatars directory not found at {avatars_path}", "avatars": []}
        
        avatar_dirs = [d.name for d in avatars_path.iterdir() if d.is_dir()]
        return {"message": f"Found {len(avatar_dirs)} avatars from {avatars_path}", "avatars": avatar_dirs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting avatar IDs: {type(e).__name__}: {str(e)}")

@app.get("/speakers", summary="Get available speaker IDs")
async def get_speaker_ids():
    try:
        index_tts_root = config.get('environments', {}).get('index-tts', {}).get('project_root')
        if not index_tts_root:
            # Assuming 'fastapi_tts' is the service name for index-tts
            index_tts_root_from_fastapitts = config.get('environments', {}).get('fastapi_tts', {}).get('project_root')
            if not index_tts_root_from_fastapitts:
                voices_path_str = resolve_path("./index-tts/voices_cache") # Fallback
            else:
                 voices_path_str = str(Path(resolve_path(index_tts_root_from_fastapitts)) / "voices_cache")
        else:
            voices_path_str = str(Path(resolve_path(index_tts_root)) / "voices_cache")


        voices_path = Path(voices_path_str)
        if not voices_path.exists() or not voices_path.is_dir():
            return {"message": f"Voices cache directory not found at {voices_path}", "speakers": []}
        
        speaker_files = [f.stem for f in voices_path.glob('*.pt') if f.is_file()]
        return {"message": f"Found {len(speaker_files)} speakers from {voices_path}", "speakers": speaker_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting speaker IDs: {type(e).__name__}: {str(e)}")


# --- 新增: 角色圖片 API 端點 ---
@app.get("/api/characters", response_model=List[CharacterInfo], summary="List all characters and their preview images")
async def list_characters(request: Request):
    characters = []
    if not BASE_CHARACTERS_DIR.is_dir():
        print(f"Character base directory not found: {BASE_CHARACTERS_DIR}")
        return []

    for char_folder in BASE_CHARACTERS_DIR.iterdir():
        if char_folder.is_dir():
            character_id = char_folder.name
            character_name = character_id.replace("_", " ").title()
            preview_image_path_in_char_folder = Path(PREVIEW_IMAGE_SUBFOLDER) / PREVIEW_IMAGE_FILENAME
            full_preview_image_path = char_folder / preview_image_path_in_char_folder
            preview_url = None
            if full_preview_image_path.is_file():
                try:
                    preview_url = str(request.url_for(
                        'get_character_image_api',
                        character_id=character_id,
                        image_path=preview_image_path_in_char_folder.as_posix()
                    ))
                except Exception as e:
                    print(f"Error generating URL for character '{character_id}' preview: {e}")
            characters.append(CharacterInfo(
                id=character_id,
                name=character_name,
                preview_image_url=preview_url
            ))
    return characters

@app.get("/api/characters/{character_id}/images/{image_path:path}",
         name="get_character_image_api",
         summary="Get a specific image for a character")
async def get_character_image_api(character_id: str, image_path: str):
    try:
        safe_character_id = Path(character_id).name
        if not safe_character_id or ".." in safe_character_id:
            raise HTTPException(status_code=400, detail="Invalid character ID.")

        character_dir_abs = (BASE_CHARACTERS_DIR / safe_character_id).resolve()

        if not str(character_dir_abs).startswith(str(BASE_CHARACTERS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Forbidden: Character path is outside base directory.")
        if not character_dir_abs.is_dir():
            raise HTTPException(status_code=404, detail=f"Character directory not found: {safe_character_id}")

        if ".." in Path(image_path).parts:
             raise HTTPException(status_code=400, detail="Invalid image path (contains '..').")

        full_image_path_abs = (character_dir_abs / image_path).resolve()

        if not str(full_image_path_abs).startswith(str(character_dir_abs)):
            raise HTTPException(status_code=403, detail="Forbidden: Image path attempts to escape character directory.")

        if not full_image_path_abs.is_file():
            raise HTTPException(status_code=404, detail=f"Image not found: {character_id}/{image_path}")

        media_type, _ = mimetypes.guess_type(full_image_path_abs)
        if media_type is None:
            media_type = "application/octet-stream"

        return FileResponse(str(full_image_path_abs), media_type=media_type)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image resource not found: {character_id}/{image_path}")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error serving image {character_id}/{image_path}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retrieving image.")

# --- Main Execution ---
if __name__ == "__main__":
    def handle_exit_signal(sig, frame):
        print(f"Manager API received signal {sig}. Initiating shutdown...")

    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    @atexit.register
    def cleanup_on_exit():
        print("atexit: Ensuring all managed servers are stopped...")
        stop_servers()

    print(f"Starting Manager API. Access via http://0.0.0.0:8200") # Port is 8070 here
    print(f"Character data is expected in: {BASE_CHARACTERS_DIR}")
    print(f"  Example character list: http://0.0.0.0:8200/api/characters")
    print(f"  Example image URL (if 'character_A' with 'full_imgs/00000001.png' exists):")
    print(f"    http://0.0.0.0:8200/api/characters/character_A/images/{PREVIEW_IMAGE_SUBFOLDER}/{PREVIEW_IMAGE_FILENAME}")

    uvicorn.run(app, host="0.0.0.0", port=8200, log_level="info") # Port is 8070 here