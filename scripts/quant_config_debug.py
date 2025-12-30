from pathlib import Path

from compressed_tensors.quantization import QuantizationConfig

config_file = (
    Path(__file__).parent.parent / "GLM-4.5-Iceblink-106B-A12B" / "config.json"
)
config = QuantizationConfig.model_validate_json(config_file.read_text())
