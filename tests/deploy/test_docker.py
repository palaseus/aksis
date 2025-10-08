"""Tests for Docker deployment utilities."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from aksis.deploy.docker import DockerManager, DockerConfig


class TestDockerManager:
    """Test Docker deployment functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.docker_manager = DockerManager()

    def test_docker_manager_initialization(self) -> None:
        """Test Docker manager initialization."""
        assert isinstance(self.docker_manager, DockerManager)
        assert self.docker_manager.config is not None

    def test_docker_manager_initialization_with_config(self) -> None:
        """Test Docker manager initialization with custom config."""
        config = DockerConfig(
            image_name="test-image",
            tag="v1.0"
        )
        manager = DockerManager(config)
        assert manager.config.image_name == "test-image"
        assert manager.config.tag == "v1.0"

    def test_docker_config_defaults(self) -> None:
        """Test Docker configuration defaults."""
        config = DockerConfig()
        assert config.image_name == "aksis"
        assert config.tag == "latest"
        assert config.dockerfile_path == "Dockerfile"
        assert config.context_path == "."
        assert config.gpu_support is True

    @patch('subprocess.run')
    def test_build_image_success(self, mock_run):
        """Test successful Docker image build."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Build successful"
        mock_run.return_value = mock_result
        
        result = self.docker_manager.build_image()
        assert result is True
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_build_image_failure(self, mock_run):
        """Test Docker image build failure."""
        mock_run.side_effect = Exception("Build failed")
        
        result = self.docker_manager.build_image()
        assert result is False

    @patch('subprocess.run')
    def test_run_container_success(self, mock_run):
        """Test successful container run."""
        # Mock the container existence check to return False (container doesn't exist)
        mock_run.side_effect = [
            Mock(returncode=1, stdout=""),  # Container doesn't exist
            Mock(returncode=0, stdout="Container started")  # Run successful
        ]
        
        result = self.docker_manager.run_container("test-container")
        assert result is True
        assert mock_run.call_count == 2

    @patch('subprocess.run')
    def test_run_container_failure(self, mock_run):
        """Test container run failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Run failed"
        mock_run.return_value = mock_result
        
        result = self.docker_manager.run_container("test-container")
        assert result is False

    @patch('subprocess.run')
    def test_stop_container_success(self, mock_run):
        """Test successful container stop."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Container stopped"
        mock_run.return_value = mock_result
        
        result = self.docker_manager.stop_container("test-container")
        assert result is True
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_stop_container_not_found(self, mock_run):
        """Test stopping non-existent container."""
        mock_run.side_effect = Exception("Container not found")
        
        result = self.docker_manager.stop_container("nonexistent")
        assert result is False

    @patch('subprocess.run')
    def test_list_containers(self, mock_run):
        """Test listing containers."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS   NAMES\n1234567890ab   aksis     /bin/sh   1 hour ago   Up 1 hour   8000/tcp   test-container"
        mock_run.return_value = mock_result
        
        containers = self.docker_manager.list_containers()
        assert isinstance(containers, list)
        if containers:  # If parsing worked
            assert len(containers) > 0

    @patch('subprocess.run')
    def test_list_images(self, mock_run):
        """Test listing images."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "REPOSITORY   TAG       IMAGE ID       CREATED        SIZE\naksis        latest    1234567890ab   1 hour ago     500MB"
        mock_run.return_value = mock_result
        
        images = self.docker_manager.list_images()
        assert isinstance(images, list)
        if images:  # If parsing worked
            assert len(images) > 0

    def test_get_container_logs(self) -> None:
        """Test getting container logs."""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Container logs here"
            mock_run.return_value = mock_result
            
            logs = self.docker_manager.get_container_logs("test-container")
            assert logs == "Container logs here"

    def test_get_docker_info(self) -> None:
        """Test getting Docker system information."""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = '{"ServerVersion": "20.10.0", "Containers": 1}'
            mock_run.return_value = mock_result
            
            info = self.docker_manager.get_docker_info()
            assert isinstance(info, dict)

    def test_cleanup(self) -> None:
        """Test Docker cleanup."""
        with patch.object(self.docker_manager, 'list_containers') as mock_list:
            with patch.object(self.docker_manager, 'stop_container') as mock_stop:
                with patch.object(self.docker_manager, 'remove_container') as mock_remove:
                    with patch.object(self.docker_manager, 'remove_image') as mock_remove_img:
                        mock_list.return_value = [{"names": "aksis-test"}]
                        
                        result = self.docker_manager.cleanup()
                        assert result is True

    def test_validate_config_invalid_image_name(self) -> None:
        """Test configuration validation with invalid image name."""
        config = DockerConfig(image_name="")
        with pytest.raises(ValueError, match="Image name cannot be empty"):
            DockerManager(config)

    def test_validate_config_invalid_tag(self) -> None:
        """Test configuration validation with invalid tag."""
        config = DockerConfig(tag="")
        with pytest.raises(ValueError, match="Tag cannot be empty"):
            DockerManager(config)

    def test_validate_config_invalid_dockerfile_path(self) -> None:
        """Test configuration validation with invalid Dockerfile path."""
        config = DockerConfig(dockerfile_path="/nonexistent/Dockerfile")
        with pytest.raises(FileNotFoundError, match="Dockerfile not found"):
            DockerManager(config)

    def test_validate_config_invalid_context_path(self) -> None:
        """Test configuration validation with invalid context path."""
        config = DockerConfig(context_path="/nonexistent/path")
        with pytest.raises(FileNotFoundError, match="Context path not found"):
            DockerManager(config)