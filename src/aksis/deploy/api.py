"""FastAPI server for chatbot inference."""

import logging
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from aksis.inference import Generator
from aksis.inference.sampler import (
    GreedySampler,
    TopKSampler,
    TopPSampler,
    TemperatureSampler,
)

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User message")
    max_new_tokens: int = Field(
        150, description="Maximum new tokens to generate"
    )
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    sampler: str = Field("temperature", description="Sampling strategy")


class GenerateRequest(BaseModel):
    """Request model for generate endpoint."""

    prompt: str = Field(..., description="Input prompt")
    max_new_tokens: int = Field(
        150, description="Maximum new tokens to generate"
    )
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    sampler: str = Field("temperature", description="Sampling strategy")
    num_sequences: int = Field(
        1, description="Number of sequences to generate"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str = Field(..., description="Generated response")
    tokens_generated: int = Field(
        ..., description="Number of tokens generated"
    )
    generation_time: float = Field(
        ..., description="Generation time in seconds"
    )


class GenerateResponse(BaseModel):
    """Response model for generate endpoint."""

    generated_text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(
        ..., description="Number of tokens generated"
    )
    generation_time: float = Field(
        ..., description="Generation time in seconds"
    )


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    uptime: float = Field(..., description="Service uptime in seconds")

    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""

    model_type: str = Field(..., description="Model type")
    vocab_size: int = Field(..., description="Vocabulary size")
    hidden_size: int = Field(..., description="Hidden size")
    num_layers: int = Field(..., description="Number of layers")
    num_heads: int = Field(..., description="Number of attention heads")

    model_config = {"protected_namespaces": ()}


class APIServer:
    """FastAPI server for chatbot inference."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize API server.

        Args:
            model_path: Path to the model checkpoint.
        """
        self.generator: Optional[Generator] = None
        self.start_time = time.time()
        self.context: List[Dict[str, str]] = []

        # Initialize FastAPI app
        self.app = FastAPI(
            title="Aksis Chatbot API",
            description="API for Aksis AI chatbot inference",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self._setup_routes()

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=self.generator is not None,
                gpu_available=self._check_gpu_availability(),
                uptime=time.time() - self.start_time,
            )

        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest) -> ChatResponse:
            """Chat endpoint for interactive conversation."""
            if not self.generator:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                start_time = time.time()

                # Add user message to context
                self.context.append(
                    {"role": "user", "content": request.message}
                )

                # Create sampler
                sampler = self._create_sampler(
                    sampler_name=request.sampler,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                )

                # Generate response
                response = self.generator.generate(
                    prompt=request.message,
                    sampler=sampler,
                    max_new_tokens=request.max_new_tokens,
                )

                generation_time = time.time() - start_time

                # Add assistant response to context
                self.context.append({"role": "assistant", "content": response})

                # Limit context size
                if len(self.context) > 20:
                    self.context = self.context[-20:]

                return ChatResponse(
                    response=response,
                    tokens_generated=len(response.split()),
                    generation_time=generation_time,
                )

            except Exception as e:
                logger.error(f"Chat generation failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Generation failed: {str(e)}"
                )

        @self.app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest) -> GenerateResponse:
            """Generate endpoint for text generation."""
            if not self.generator:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                start_time = time.time()

                # Create sampler
                sampler = self._create_sampler(
                    sampler_name=request.sampler,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                )

                # Generate text
                generated_text = self.generator.generate(
                    prompt=request.prompt,
                    sampler=sampler,
                    max_new_tokens=request.max_new_tokens,
                )

                generation_time = time.time() - start_time

                return GenerateResponse(
                    generated_text=generated_text,
                    tokens_generated=len(generated_text.split()),
                    generation_time=generation_time,
                )

            except Exception as e:
                logger.error(f"Text generation failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Generation failed: {str(e)}"
                )

        @self.app.get("/model/info", response_model=ModelInfoResponse)
        async def model_info() -> ModelInfoResponse:
            """Get model information."""
            if not self.generator:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                info = self.generator.get_model_info()
                return ModelInfoResponse(**info)

            except Exception as e:
                logger.error(f"Failed to get model info: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get model info: {str(e)}",
                )

        @self.app.get("/model_info", response_model=ModelInfoResponse)
        async def model_info_alt() -> ModelInfoResponse:
            """Alternative model information endpoint."""
            if not self.generator:
                raise HTTPException(status_code=503, detail="Model not loaded")

            try:
                info = self.generator.get_model_info()
                return ModelInfoResponse(**info)

            except Exception as e:
                logger.error(f"Failed to get model info: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get model info: {str(e)}",
                )

        @self.app.get("/metrics")
        async def metrics() -> str:
            """Prometheus metrics endpoint."""
            # This would integrate with the monitoring system
            from fastapi import Response

            return Response(
                content="# Prometheus metrics would be here\n",
                media_type="text/plain",
            )

        @self.app.websocket("/ws/chat")
        async def websocket_chat(websocket: WebSocket) -> None:
            """WebSocket endpoint for real-time chat."""
            await websocket.accept()

            try:
                while True:
                    # Receive message
                    data = await websocket.receive_json()
                    message = data.get("message", "")

                    if not message:
                        await websocket.send_json({"error": "Empty message"})
                        continue

                    if not self.generator:
                        await websocket.send_json(
                            {"error": "Model not loaded"}
                        )
                        continue

                    try:
                        # Generate response
                        response = self.generator.generate(
                            prompt=message, max_new_tokens=150, temperature=0.7
                        )

                        # Send response
                        await websocket.send_json({"response": response})

                    except Exception as e:
                        logger.error(f"WebSocket generation failed: {e}")
                        await websocket.send_json({"error": str(e)})

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({"error": str(e)})

    def load_model(self, model_path: str) -> bool:
        """Load model from checkpoint.

        Args:
            model_path: Path to the model checkpoint.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            if not model_path:
                logger.warning("No model path provided")
                return False

            logger.info(f"Loading model from {model_path}")

            # Create tokenizer and build vocabulary to match training
            from aksis.data.tokenizer import Tokenizer
            from aksis.train.dataset import load_wikitext2

            tokenizer = Tokenizer(vocab_size=10000)

            # Load WikiText-2 dataset to build the same vocabulary used in training
            try:
                train_dataset, _, _ = load_wikitext2(tokenizer, max_length=512)
                logger.info(
                    f"Vocabulary built with {tokenizer.vocab_size_with_special} tokens"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load WikiText-2 for vocabulary: {e}"
                )
                # Fallback to sample texts
                sample_texts = [
                    "Hello world, this is a test sentence.",
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning is a subset of artificial intelligence.",
                    "Natural language processing helps computers understand human language.",
                    "Deep learning uses neural networks with multiple layers.",
                ]
                tokenizer.build_vocab(sample_texts)

            # Load generator from checkpoint
            self.generator = Generator.load_from_checkpoint(
                checkpoint_path=model_path,
                tokenizer=tokenizer,
                device=None,  # Will auto-detect CUDA
                max_length=512,
                use_mixed_precision=False,
            )

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.generator = None
            return False

    def _create_sampler(
        self,
        sampler_name: str,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        """Create a sampler object based on the sampler name.

        Args:
            sampler_name: Name of the sampler to create.
            temperature: Temperature for temperature sampling.
            top_k: Top-k parameter for top-k sampling.
            top_p: Top-p parameter for top-p sampling.

        Returns:
            Sampler object.
        """
        if sampler_name == "greedy":
            return GreedySampler()
        elif sampler_name == "top-k":
            return TopKSampler(k=top_k)
        elif sampler_name == "top-p":
            return TopPSampler(p=top_p)
        elif sampler_name == "temperature":
            return TemperatureSampler(temperature=temperature)
        else:
            # Default to greedy
            return GreedySampler()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available.

        Returns:
            True if GPU is available, False otherwise.
        """
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        reload: bool = False,
    ) -> None:
        """Run the API server.

        Args:
            host: Host to bind to.
            port: Port to bind to.
            workers: Number of worker processes.
            reload: Whether to enable auto-reload.
        """
        logger.info(f"Starting API server on {host}:{port}")

        uvicorn.run(
            self.app, host=host, port=port, workers=workers, reload=reload
        )
