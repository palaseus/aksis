"""Docker utilities for deployment."""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DockerConfig:
    """Configuration for Docker operations."""
    
    image_name: str = "aksis"
    tag: str = "latest"
    dockerfile_path: str = "Dockerfile"
    context_path: str = "."
    build_args: Dict[str, str] = None
    ports: Dict[str, str] = None
    volumes: Dict[str, str] = None
    environment: Dict[str, str] = None
    gpu_support: bool = True
    cuda_version: str = "11.8"
    
    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.build_args is None:
            self.build_args = {}
        if self.ports is None:
            self.ports = {"8000": "8000"}
        if self.volumes is None:
            self.volumes = {}
        if self.environment is None:
            self.environment = {}


class DockerManager:
    """Manager for Docker operations."""
    
    def __init__(self, config: Optional[DockerConfig] = None) -> None:
        """Initialize Docker manager.
        
        Args:
            config: Docker configuration. If None, uses default config.
        """
        self.config = config or DockerConfig()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate Docker configuration."""
        if not self.config.image_name:
            raise ValueError("Image name cannot be empty")
        
        if not self.config.tag:
            raise ValueError("Tag cannot be empty")
        
        # Only validate file existence if they're not default paths
        if self.config.dockerfile_path != "Dockerfile" and not os.path.exists(self.config.dockerfile_path):
            raise FileNotFoundError(f"Dockerfile not found: {self.config.dockerfile_path}")
        
        if self.config.context_path != "." and not os.path.exists(self.config.context_path):
            raise FileNotFoundError(f"Context path not found: {self.config.context_path}")
    
    def build_image(self, force_rebuild: bool = False) -> bool:
        """Build Docker image.
        
        Args:
            force_rebuild: Whether to force rebuild even if image exists.
            
        Returns:
            True if build successful, False otherwise.
        """
        try:
            # Check if image already exists
            if not force_rebuild and self._image_exists():
                logger.info(f"Image {self.config.image_name}:{self.config.tag} already exists")
                return True
            
            # Build command
            cmd = [
                "docker", "build",
                "-t", f"{self.config.image_name}:{self.config.tag}",
                "-f", self.config.dockerfile_path,
                self.config.context_path
            ]
            
            # Add build args
            for key, value in self.config.build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])
            
            logger.info(f"Building Docker image: {' '.join(cmd)}")
            
            # Execute build
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Docker image built successfully")
            logger.debug(f"Build output: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Docker build: {e}")
            return False
    
    def run_container(
        self,
        container_name: Optional[str] = None,
        detach: bool = True,
        remove: bool = False
    ) -> bool:
        """Run Docker container.
        
        Args:
            container_name: Name for the container. If None, auto-generates.
            detach: Whether to run in detached mode.
            remove: Whether to remove container after it stops.
            
        Returns:
            True if run successful, False otherwise.
        """
        try:
            if container_name is None:
                container_name = f"{self.config.image_name}-{self.config.tag}"
            
            # Check if container already exists
            if self._container_exists(container_name):
                logger.warning(f"Container {container_name} already exists")
                return False
            
            # Run command
            cmd = [
                "docker", "run",
                "--name", container_name
            ]
            
            # Add detach flag
            if detach:
                cmd.append("-d")
            
            # Add remove flag
            if remove:
                cmd.append("--rm")
            
            # Add GPU support
            if self.config.gpu_support:
                cmd.extend(["--gpus", "all"])
            
            # Add ports
            for host_port, container_port in self.config.ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])
            
            # Add volumes
            for host_path, container_path in self.config.volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])
            
            # Add environment variables
            for key, value in self.config.environment.items():
                cmd.extend(["-e", f"{key}={value}"])
            
            # Add image
            cmd.append(f"{self.config.image_name}:{self.config.tag}")
            
            logger.info(f"Running Docker container: {' '.join(cmd)}")
            
            # Execute run
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Docker container {container_name} started successfully")
            logger.debug(f"Run output: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker run failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Docker run: {e}")
            return False
    
    def stop_container(self, container_name: str) -> bool:
        """Stop Docker container.
        
        Args:
            container_name: Name of the container to stop.
            
        Returns:
            True if stop successful, False otherwise.
        """
        try:
            cmd = ["docker", "stop", container_name]
            
            logger.info(f"Stopping Docker container: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Docker container {container_name} stopped successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker stop failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Docker stop: {e}")
            return False
    
    def remove_container(self, container_name: str) -> bool:
        """Remove Docker container.
        
        Args:
            container_name: Name of the container to remove.
            
        Returns:
            True if remove successful, False otherwise.
        """
        try:
            cmd = ["docker", "rm", container_name]
            
            logger.info(f"Removing Docker container: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Docker container {container_name} removed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker remove failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Docker remove: {e}")
            return False
    
    def remove_image(self, image_name: Optional[str] = None, tag: Optional[str] = None) -> bool:
        """Remove Docker image.
        
        Args:
            image_name: Name of the image to remove. If None, uses config.
            tag: Tag of the image to remove. If None, uses config.
            
        Returns:
            True if remove successful, False otherwise.
        """
        try:
            if image_name is None:
                image_name = self.config.image_name
            if tag is None:
                tag = self.config.tag
            
            cmd = ["docker", "rmi", f"{image_name}:{tag}"]
            
            logger.info(f"Removing Docker image: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Docker image {image_name}:{tag} removed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker remove image failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Docker remove image: {e}")
            return False
    
    def list_containers(self, all_containers: bool = False) -> List[Dict[str, str]]:
        """List Docker containers.
        
        Args:
            all_containers: Whether to include stopped containers.
            
        Returns:
            List of container information dictionaries.
        """
        try:
            cmd = ["docker", "ps"]
            if all_containers:
                cmd.append("-a")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 1:  # Only header
                return []
            
            containers = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 6:
                    containers.append({
                        "container_id": parts[0],
                        "image": parts[1],
                        "command": parts[2],
                        "created": parts[3],
                        "status": parts[4],
                        "ports": parts[5] if len(parts) > 5 else "",
                        "names": parts[6] if len(parts) > 6 else ""
                    })
            
            return containers
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker list containers failed: {e.stderr}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during Docker list containers: {e}")
            return []
    
    def list_images(self) -> List[Dict[str, str]]:
        """List Docker images.
        
        Returns:
            List of image information dictionaries.
        """
        try:
            cmd = ["docker", "images"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 1:  # Only header
                return []
            
            images = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 5:
                    images.append({
                        "repository": parts[0],
                        "tag": parts[1],
                        "image_id": parts[2],
                        "created": parts[3],
                        "size": parts[4]
                    })
            
            return images
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker list images failed: {e.stderr}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during Docker list images: {e}")
            return []
    
    def get_container_logs(self, container_name: str, tail: int = 100) -> str:
        """Get container logs.
        
        Args:
            container_name: Name of the container.
            tail: Number of lines to return.
            
        Returns:
            Container logs as string.
        """
        try:
            cmd = ["docker", "logs", "--tail", str(tail), container_name]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker logs failed: {e.stderr}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error during Docker logs: {e}")
            return ""
    
    def _image_exists(self) -> bool:
        """Check if Docker image exists.
        
        Returns:
            True if image exists, False otherwise.
        """
        try:
            cmd = ["docker", "images", "-q", f"{self.config.image_name}:{self.config.tag}"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            return bool(result.stdout.strip())
            
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False
    
    def _container_exists(self, container_name: str) -> bool:
        """Check if Docker container exists.
        
        Args:
            container_name: Name of the container.
            
        Returns:
            True if container exists, False otherwise.
        """
        try:
            cmd = ["docker", "ps", "-a", "-q", "-f", f"name={container_name}"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            return bool(result.stdout.strip())
            
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False
    
    def get_docker_info(self) -> Dict[str, Union[str, int, bool]]:
        """Get Docker system information.
        
        Returns:
            Dictionary containing Docker system information.
        """
        try:
            cmd = ["docker", "info", "--format", "{{json .}}"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            import json
            return json.loads(result.stdout)
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Docker info failed: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error during Docker info: {e}")
            return {}
    
    def cleanup(self) -> bool:
        """Clean up Docker resources.
        
        Returns:
            True if cleanup successful, False otherwise.
        """
        try:
            # Stop and remove containers
            containers = self.list_containers(all_containers=True)
            for container in containers:
                if container["names"].startswith(self.config.image_name):
                    self.stop_container(container["names"])
                    self.remove_container(container["names"])
            
            # Remove image
            self.remove_image()
            
            logger.info("Docker cleanup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Docker cleanup failed: {e}")
            return False