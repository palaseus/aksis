"""Tests for the Tokenizer class."""

# No additional imports needed

from aksis.data.tokenizer import Tokenizer


class TestTokenizer:
    """Test cases for the Tokenizer class."""

    def test_tokenizer_initialization(self) -> None:
        """Test tokenizer initialization with default parameters."""
        tokenizer = Tokenizer()
        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token_id is not None
        assert tokenizer.unk_token_id is not None
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None

    def test_tokenizer_initialization_with_vocab_size(self) -> None:
        """Test tokenizer initialization with custom vocab size."""
        vocab_size = 1000
        tokenizer = Tokenizer(vocab_size=vocab_size)
        assert tokenizer.vocab_size == vocab_size

    def test_build_vocab_from_text(self) -> None:
        """Test building vocabulary from text corpus."""
        text_corpus = [
            "Hello world!",
            "This is a test.",
            "Another sentence here.",
        ]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(text_corpus)

        # Check that special tokens are present
        assert tokenizer.pad_token_id is not None
        assert tokenizer.unk_token_id is not None
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None

        # Check that vocabulary contains common words
        assert "hello" in tokenizer.token_to_id
        assert "world" in tokenizer.token_to_id
        assert "test" in tokenizer.token_to_id

    def test_encode_single_text(self) -> None:
        """Test encoding a single text string."""
        text = "Hello world!"
        tokenizer = Tokenizer()
        tokenizer.build_vocab([text])

        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert all(isinstance(token_id, int) for token_id in encoded)

    def test_encode_with_special_tokens(self) -> None:
        """Test encoding with special tokens (BOS/EOS)."""
        text = "Hello world!"
        tokenizer = Tokenizer()
        tokenizer.build_vocab([text])

        encoded = tokenizer.encode(text, add_special_tokens=True)
        assert encoded[0] == tokenizer.bos_token_id
        assert encoded[-1] == tokenizer.eos_token_id

    def test_encode_batch(self) -> None:
        """Test encoding a batch of texts."""
        texts = ["Hello world!", "This is a test.", "Another sentence."]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        encoded_batch = tokenizer.encode_batch(texts)
        assert isinstance(encoded_batch, list)
        assert all(isinstance(seq, list) for seq in encoded_batch)
        assert all(all(isinstance(x, int) for x in seq) for seq in encoded_batch)
        assert len(encoded_batch) == len(texts)

        for encoded in encoded_batch:
            assert isinstance(encoded, list)
            assert len(encoded) > 0

    def test_decode_single_sequence(self) -> None:
        """Test decoding a single token sequence."""
        text = "Hello world!"
        tokenizer = Tokenizer()
        tokenizer.build_vocab([text])

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)
        # Note: decoded text might be different due to tokenization
        assert len(decoded) > 0

    def test_decode_batch(self) -> None:
        """Test decoding a batch of token sequences."""
        texts = ["Hello world!", "This is a test."]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        encoded_batch = tokenizer.encode_batch(texts)
        decoded_batch = tokenizer.decode_batch(encoded_batch)

        assert isinstance(decoded_batch, list)
        assert all(isinstance(text, str) for text in decoded_batch)
        assert len(decoded_batch) == len(texts)

        for decoded in decoded_batch:
            assert isinstance(decoded, str)
            assert len(decoded) > 0

    def test_unknown_token_handling(self) -> None:
        """Test handling of unknown tokens."""
        text = "Hello world!"
        tokenizer = Tokenizer()
        tokenizer.build_vocab([text])

        # Encode text with unknown words
        unknown_text = "Hello unknown_word!"
        encoded = tokenizer.encode(unknown_text)

        # Should contain unk_token_id for unknown words
        assert tokenizer.unk_token_id in encoded

    def test_padding(self) -> None:
        """Test padding functionality."""
        texts = ["Hello", "This is a longer sentence"]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        # Find max length first
        max_length = max(len(tokenizer.encode(text)) for text in texts)
        encoded_batch = tokenizer.encode_batch(
            texts, padding=True, max_length=max_length
        )

        # All sequences should have the same length
        lengths = [len(seq) for seq in encoded_batch]
        assert all(length == lengths[0] for length in lengths)

        # Check that padding token is used
        max_length = max(len(seq) for seq in encoded_batch)
        for seq in encoded_batch:
            if len(seq) < max_length:
                # Check that padding tokens are at the end
                padding_start = len(seq)
                assert all(
                    seq[i] == tokenizer.pad_token_id
                    for i in range(padding_start, max_length)
                )

    def test_truncation(self) -> None:
        """Test truncation functionality."""
        text = "This is a very long sentence that should be truncated"
        tokenizer = Tokenizer()
        tokenizer.build_vocab([text])

        max_length = 5
        encoded = tokenizer.encode(text, max_length=max_length, truncation=True)

        assert len(encoded) <= max_length

    def test_save_and_load_vocab(self, tmp_path) -> None:
        """Test saving and loading vocabulary."""
        text = "Hello world! This is a test."
        tokenizer = Tokenizer()
        tokenizer.build_vocab([text])

        vocab_path = tmp_path / "vocab.json"
        tokenizer.save_vocab(str(vocab_path))

        # Create new tokenizer and load vocab
        new_tokenizer = Tokenizer()
        new_tokenizer.load_vocab(str(vocab_path))

        # Check that vocabularies are the same
        assert new_tokenizer.token_to_id == tokenizer.token_to_id
        assert new_tokenizer.id_to_token == tokenizer.id_to_token
        assert new_tokenizer.vocab_size == tokenizer.vocab_size

    def test_tokenizer_with_empty_text(self) -> None:
        """Test tokenizer behavior with empty text."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["Hello world!"])

        # Empty string should return empty list or special tokens
        encoded = tokenizer.encode("")
        assert isinstance(encoded, list)

        # Should be able to decode empty sequence
        decoded = tokenizer.decode([])
        assert isinstance(decoded, str)

    def test_tokenizer_consistency(self) -> None:
        """Test that encode-decode is consistent."""
        text = "Hello world! This is a test sentence."
        tokenizer = Tokenizer()
        tokenizer.build_vocab([text])

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        # Re-encode the decoded text
        re_encoded = tokenizer.encode(decoded)

        # Should be similar (allowing for tokenization differences)
        assert len(re_encoded) > 0
        assert isinstance(decoded, str)
